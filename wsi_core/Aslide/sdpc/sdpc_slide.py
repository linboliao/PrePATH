import numpy.ctypeslib as npCtypes
import ctypes
from ctypes import *
import gc
import os
import sys
import numpy as np
from PIL import Image
from .Sdpc_struct import SqSdpcInfo


# import dll file
dirname, _ = os.path.split(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, 'so'))
soPath = os.path.join(dirname, 'so/libDecodeSdpc.so')

# load dll
so = ctypes.CDLL(soPath)
so.GetLayerInfo.restype = POINTER(c_char)
so.SqGetRoiRgbOfSpecifyLayer.argtypes = [POINTER(SqSdpcInfo), POINTER(POINTER(c_uint8)),
                                             c_int, c_int, c_uint, c_uint, c_int]
so.SqGetRoiRgbOfSpecifyLayer.restype = c_int
so.SqOpenSdpc.restype = POINTER(SqSdpcInfo)
so.GetLabelJpeg.argtypes = [POINTER(SqSdpcInfo), POINTER(c_uint), POINTER(c_uint), POINTER(c_size_t)]
so.GetLabelJpeg.restype = POINTER(c_uint8)


class SdpcSlide:

    def __init__(self, sdpcPath):
        self.sdpc = self.readSdpc(sdpcPath)
        self.level_count = self.getLevelCount()
        self.level_downsamples = self.getLevelDownsamples()
        self.level_dimensions = self.getLevelDimensions()
        self.scan_magnification = self.readSdpc(sdpcPath).contents.picHead.contents.rate
        self.sampling_rate = self.readSdpc(sdpcPath).contents.picHead.contents.scale
        self.properties = {'openslide.mpp-t': self.sampling_rate, 'openslide.mpp-x': self.sampling_rate, 'openslide.vendor': 'TEKSQRAY'} # maintain consistency with openslide API

    @property
    def dimensions(self):
        """Return the dimensions of the highest resolution level (level 0)."""
        return self.level_dimensions[0] if self.level_dimensions else (0, 0)

    def getRgb(self, rgbPos, width, height):

        intValue = npCtypes.as_array(rgbPos, (height, width, 3))
        return intValue

    def readSdpc(self, fileName):

        sdpc = so.SqOpenSdpc(c_char_p(bytes(fileName, 'utf-8')))
        sdpc.contents.fileName = bytes(fileName, 'utf-8')

        return sdpc

    def getLevelCount(self):

        return self.sdpc.contents.picHead.contents.hierarchy

    def getLevelDownsamples(self):

        levelCount = self.getLevelCount()
        rate = self.sdpc.contents.picHead.contents.scale
        rate = 1 / rate
        _list = []
        for i in range(levelCount):
            _list.append(rate ** i)
        return tuple(_list)

    def get_best_level_for_downsample(self, downsample):

        preset = [i*i for i in self.level_downsamples]
        err = [abs(i-downsample) for i in preset]
        level = err.index(min(err))
        return level

    def read_region(self, location, level, size):

        startX, startY = location
        scale = self.level_downsamples[level]
        startX = int(startX / scale)
        startY = int(startY / scale)

        width, height = size

        rgbPos = POINTER(c_uint8)()
        rgbPosPointer = byref(rgbPos)
        try:
            result = so.SqGetRoiRgbOfSpecifyLayer(self.sdpc, rgbPosPointer, width, height, startX, startY, level)
            if result != 0:
                # if result != 0, raise an exception
                raise Exception("Failed to read region")

            rgb = self.getRgb(rgbPos, width, height)[..., ::-1]
            rgbCopy = rgb.copy()

            img = Image.fromarray(rgbCopy)
            return img
        finally:
            # Ensure resources are released
            if rgbPos:
                so.Dispose(rgbPos)

            del rgbPos
            del rgbPosPointer

            gc.collect()

    def get_thumbnail(self, thumbnail_size):
        thumbnail = self.read_region((0, 0), len(self.level_dimensions) - 1, self.level_dimensions[-1])
        thumbnail = thumbnail.resize(thumbnail_size)
        return thumbnail

    def getLevelDimensions(self):

        def findStrIndex(subStr, str):
            index1 = str.find(subStr)
            index2 = str.find(subStr, index1 + 1)
            index3 = str.find(subStr, index2 + 1)
            index4 = str.find(subStr, index3 + 1)
            return index1, index2, index3, index4

        levelCount = self.getLevelCount()
        levelDimensions = []
        for level in range(levelCount):
            layerInfo = so.GetLayerInfo(self.sdpc, level)
            try:
                count = 0
                byteList = []
                while (ord(layerInfo[count]) != 0):
                    byteList.append(layerInfo[count])
                    count += 1

                strList = [byteValue.decode('utf-8') for byteValue in byteList]
                str = ''.join(strList)

                equal1, equal2, equal3, equal4 = findStrIndex("=", str)
                line1, line2, line3, line4 = findStrIndex("|", str)

                rawWidth = int(str[equal1 + 1:line1])
                rawHeight = int(str[equal2 + 1:line2])
                boundWidth = int(str[equal3 + 1:line3])
                boundHeight = int(str[equal4 + 1:line4])
                w, h = rawWidth - boundWidth, rawHeight - boundHeight
                levelDimensions.append((w, h))
            finally:
                # 释放 layerInfo 资源
                if hasattr(so, 'FreeLayerInfo'):
                    so.FreeLayerInfo(layerInfo)
                elif hasattr(so, 'Dispose'):
                    so.Dispose(layerInfo)

        return tuple(levelDimensions)

    def saveLabelImg(self):
        wPos = POINTER(c_uint)(c_uint(0))
        hPos = POINTER(c_uint)(c_uint(0))
        sizePos = POINTER(c_size_t)(c_size_t(0))
        rgb_pos = so.GetLabelJpeg(self.sdpc, wPos, hPos, sizePos)
        save_path = './cache/label.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'bw') as f:
            buf = bytearray(rgb_pos[:sizePos.contents.value])
            f.write(buf)
        f.close()
        label_img = Image.open(save_path)
        # remove the temporary file
        if os.path.exists(save_path):
            os.remove(save_path)
        return label_img

    def close(self):
        try:
            if hasattr(self, 'sdpc') and self.sdpc:
                so.SqCloseSdpc(self.sdpc)
                self.sdpc = None
        except Exception as e:
            print(f"Error closing SDPC file: {e}")
        finally:
            # 强制清理内存
            gc.collect()