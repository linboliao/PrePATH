#patch_size=224
#save_dir=/NAS145/Data/治疗/patches_0_$patch_size
#wsi_dir=/NAS145/Data/治疗/slides
#wsi_format="svs"
#
#python create_patches_fp.py --source $wsi_dir --save_dir $save_dir --patch_size $patch_size --step_size $patch_size --preset xiehe.csv --patch_level 0 --wsi_format $wsi_format --seg --patch --stitch --use_mp
cd ../../

patch_size=224
patch_level=0
save_dir=/NAS145/Data/MXB/CLS/patches_0_$patch_size
wsi_dir=/NAS145/Data/MXB/CLS
wsi_format="svs;kfb"

python create_patches_fp_mag.py --source $wsi_dir --save_dir $save_dir --patch_size $patch_size --step_size $patch_size --preset xiehe.csv --patch_level $patch_level --wsi_format $wsi_format --seg --patch --stitch --use_mp