cd ../../

patch_size=448

save_dir="/NAS2/Data1/lbliao/Data/MXB/classification/第一批/patches_0_$patch_size"
wsi_dir="/NAS2/Data1/lbliao/Data/MXB/classification/第一批/slides"
wsi_format="kfb;svs"

python create_patches_fp.py --source $wsi_dir --save_dir $save_dir --patch_size $patch_size --step_size $patch_size --preset maixin.csv --patch_level 0 --wsi_format $wsi_format --seg --patch --stitch --use_mp --no_auto_skip
