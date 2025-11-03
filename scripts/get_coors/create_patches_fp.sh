
save_dir=/NAS3/lbliao/Data/CRC/分子检测
wsi_dir=/NAS2/lbliao/CRC分子检测
wsi_format="ndpi"
patch_size=448

python create_patches_fp.py --source $wsi_dir --save_dir $save_dir --patch_size $patch_size --step_size $patch_size --preset xiehe.csv --patch_level 0 --wsi_format $wsi_format --seg --patch --stitch --use_mp