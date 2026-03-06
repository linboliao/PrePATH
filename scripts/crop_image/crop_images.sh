# configuration
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

wsi_dir=/NAS4/llb/协和医院结直肠癌数据
wsi_format='svs'
coord=/NAS145/Data/CRC/协和/patches_tumor_0_224
h5_dir=$coord/patches
save_dir=/NAS145/Data/CRC/协和/images_tumor_0_224
datatype="auto"
level=0
size=224
cpu_cores=20
#cd ../../
python extract_images.py --datatype $datatype --wsi_format $wsi_format --level $level --size $size --cpu_cores $cpu_cores --h5_root $h5_dir --save_root $save_dir --wsi_root $wsi_dir
