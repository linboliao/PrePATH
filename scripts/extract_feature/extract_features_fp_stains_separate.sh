export PYTHONPATH=../../PrePATH:$PYTHONPATH
export LD_LIBRARY_PATH=/data12/jing/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=5 python extract_features_fp_stains_separate.py \
--data_coors_dir /NAS3/lbliao/Data/CRC/协和/cls/patches \
--data_slide_dir /NAS4/llb/协和医院结直肠癌数据/slides --slide_ext '.ndpi;.kfb' \
--csv_path csv/extract_features_s_s/part_1.csv \
--feat_dir /NAS3/lbliao/Data/CRC/协和/cls/feat/test