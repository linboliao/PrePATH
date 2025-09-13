export PYTHONPATH=../../PrePATH:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
coors_dir=/NAS2/Data1/lbliao/Data/CRC/浙肿
wsi_dir=/NAS2/Data4/llb/浙江省肿瘤医院结直肠癌数据
feat_dir=/NAS2/Data1/lbliao/Data/CRC/浙肿/tumor_feat
csv_path=csv/extract_features_tumor_zz
ckpt=/data2/lbliao/Code/MIL_BASELINE/preprocess/best_model.pth

cd ../../
#python scripts/extract_feature/generate_csv.py --h5_dir $coors_dir/patches --num 2 --root $csv_path
CUDA_VISIBLE_DEVICES=1 python extract_features_fp_tumor.py --data_coors_dir $coors_dir --data_slide_dir $wsi_dir --slide_ext '.kfb' --csv_path $csv_path/part_1.csv --feat_dir $feat_dir --model uni --ckpt $ckpt