# The coordinates of the patches are saved, you can change the path to any directory you want.
save_dir="/NAS3/lbliao/Data/CRC/协和/cls"
# The directory where the WSI files are stored, you can change it to any directory you want.
wsi_dir="/NAS4/llb/协和医院结直肠癌数据/slides"
# The WSI format, you can set it based on the format of your WSI
label_dir="/NAS3/lbliao/Data/CRC/协和/cls/geojson/normal"

# Normally, you don't need to change following lines.
# to set the patch size, please set it at `configs/resolution.py`
python create_patches_tumor.py --wsi_dir $wsi_dir --save_dir $save_dir --label_dir $label_dir --vis