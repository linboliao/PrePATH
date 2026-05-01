cd ../../

patch_size=896
patch_level=0
save_dir=/NAS145/liaolinbo/Data/MXB/test/patches_0_$patch_size
wsi_dir=/NAS145/liaolinbo/Data/MXB/test/slides
wsi_format="svs;kfb;tif;tiff"

python create_patches_fp_mag.py --source $wsi_dir --save_dir $save_dir --patch_size $patch_size --step_size $patch_size --preset ihc.csv --patch_level $patch_level --wsi_format $wsi_format --seg --patch --stitch --use_mp