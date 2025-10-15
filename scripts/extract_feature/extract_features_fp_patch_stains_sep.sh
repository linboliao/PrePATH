export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/data12/jing/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
model=h-optimus-1
patch_img_dir=/NAS4/llb/Data/cls/NCT-CRC-HE_含数据集介绍/CRC-VAL-HE-7K # NCT-CRC-HE-100K CRC-VAL-HE-7K
feat_dir=/NAS3/lbliao/Data/CRC/协和/cls/nct_feat/$model/val1 # train test
CUDA_VISIBLE_DEVICES=3 python extract_features_fp_patch_stains_sep.py --patch_img_dir $patch_img_dir --feat_dir $feat_dir --model $model #--mode val