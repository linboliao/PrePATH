#cd ../../

patch_size=224
patch_level=0
save_dir=/NAS145/Data/MXB/外部测试/patches_0_$patch_size
wsi_dir=/NAS145/Data/MXB/外部测试
wsi_format="svs;kfb;tif;tiff"

python create_patches_fp.py --source $wsi_dir --save_dir $save_dir --patch_size $patch_size --step_size $patch_size --preset ihc.csv --patch_level $patch_level --wsi_format $wsi_format --seg --patch --stitch --use_mp