# configuration
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
wsi_dir=/NAS4/llb/协和医院结直肠癌数据/slides
wsi_format=ndpi
log_path=scripts/crop_image/logs
coor_root=/NAS3/lbliao/Data/CRC/协和/patches_1
save_dir=/NAS3/lbliao/Data/CRC/协和/tumor_image
datatype="auto"
level=0
size=512
cpu_cores=50


h5_dir=$coor_root"/patches"

python extract_images.py \
        --datatype $datatype \
        --wsi_format $wsi_format \
        --level $level \
        --size $size \
        --cpu_cores $cpu_cores \
        --h5_root $h5_dir \
        --save_root $save_dir \
        --wsi_root $wsi_dir > crop_img.log
