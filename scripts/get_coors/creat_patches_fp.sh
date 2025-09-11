# The coordinates of the patches are saved, you can change the path to any directory you want.
save_dir="/NAS2/Data1/lbliao/Data/MXB/classification/测试一/yolo_patches"
# The directory where the WSI files are stored, you can change it to any directory you want.
wsi_dir="/NAS2/Data1/lbliao/Data/MXB/classification/测试一/slides"
# The WSI format, you can set it based on the format of your WSI
wsi_format="kfb"
# The log file name, you can change it to any name you want.
log_name="SAL.log"
cd ../../
# Normally, you don't need to change following lines.
# to set the patch size, please set it at `configs/resolution.py`
python create_patches_fp.py --source $wsi_dir --save_dir $save_dir --patch_size 2048 --step_size 2048 --preset maixin.csv --patch_level 0 --wsi_format $wsi_format --seg --patch --stitch --use_mp --no_auto_skip