export PYTHONPATH=../../PrePATH:$PYTHONPATH
export LD_LIBRARY_PATH=~/miniconda3/envs/PrePATH/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
coors_dir=/NAS3/lbliao/Data/CRC/中日
wsi_dir=/NAS4/llb/中日友好医院结直肠癌数据/slides
feat_dir=/NAS3/lbliao/Data/CRC/中日/tumor_feats
csv_path=csv/extract_features_tumor_zr
ckpt=/NAS3/lbliao/Code/MIL_BASELINE/preprocess/nct_best_model.pth
slide_ext=".ndpi;.kfb;.svs"

python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 2 --root $csv_path
CUDA_VISIBLE_DEVICES=0 python extract_features_fp_tumor.py --batch_size 128 --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --csv_path $csv_path/part_0.csv --feat_dir $feat_dir --model uni --ckpt $ckpt
#echo --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext '.kfb' --csv_path $csv_path/part_2.csv --feat_dir $feat_dir --model uni --ckpt $ckpt