export PYTHONPATH=../../PrePATH:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
coord_dir=/NAS2/Data1/lbliao/Data/CRC/协和/cls
wsi_dir=/NAS2/Data4/llb/协和医院结直肠癌数据/slides
csv_path=csv/extract_features_s_s
slide_ext='.ndpi;.kfb'
feat_dir=/NAS2/Data1/lbliao/Data/CRC/协和/cls/feats/train
cd ../../
CUDA_VISIBLE_DEVICES=2 python extract_features_fp_stains_separate.py --data_coors_dir $coord_dir --data_slide_dir $wsi_dir --slide_ext $slide_ext --csv_path $csv_path/part_6.csv --feat_dir $feat_dir