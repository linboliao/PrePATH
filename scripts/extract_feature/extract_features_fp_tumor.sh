export PYTHONPATH=../../PrePATH:$PYTHONPATH
export LD_LIBRARY_PATH=/data12/jing/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
coors_dir=/NAS3/lbliao/Data/CRC/协和/
wsi_dir=/NAS4/llb/协和医院结直肠癌数据/slides
feat_dir=/NAS3/lbliao/Data/CRC/协和/tumor_feat
csv_path=csv/extract_features_tumor
ckpt=/NAS3/lbliao/Code/MIL_BASELINE/preprocess/nct_best_model.pth
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 5 --root $csv_path

CUDA_VISIBLE_DEVICES=3 python extract_features_fp_tumor.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext '.ndpi;.kfb' --csv_path $csv_path/part_2.csv --feat_dir $feat_dir --model uni --ckpt $ckpt