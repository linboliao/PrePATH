export PYTHONPATH=../../PrePATH:$PYTHONPATH
export LD_LIBRARY_PATH=/data12/jing/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
coors_dir=/NAS3/lbliao/Data/CRC/协和/patches
wsi_dir=/NAS4/llb/协和医院结直肠癌数据/slides
feat_dir=/NAS3/lbliao/Data/CRC/协和/tumor_feat
csv_path=csv/extract_features_tumor
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 5 --root $csv_path

CUDA_VISIBLE_DEVICES=4 python extract_features_fp_tumor.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext '.ndpi;.kfb' --csv_path $csv_path/part_1.csv --feat_dir $feat_dir --model uni