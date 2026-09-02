"""
configure the image size under specific resolution.
The key means the maximum resolution of the WSI.
The value means the images size we cropped on that resolution.
We set this try to reduce the influence of resolution.
"""

STEPS = {"20x": 256, "40x": 512, "80x": 1024}
SIZES = {"20x": 256, "40x": 512, "80x": 1024}
# Reference microns-per-pixel that every slide is resampled to during
# patching, so the encoder always sees tissue at one physical scale
# regardless of scan magnification. ~0.5 um/px == "20x".
TARGET_MPP = 0.5
