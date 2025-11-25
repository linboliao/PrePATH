# configuration
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

level=0
size=448
cpu_cores=20
#wsi_dir=/NAS4/llb/中日友好医院结直肠癌数据/slides
#wsi_format="kfb;svs"
#h5_dir=/NAS145/Data/LS/zr/patches_0_$size/patches
#save_dir=/NAS145/Data/LS/zr/image
datatype="auto"
#
#python extract_images.py --datatype $datatype --wsi_format $wsi_format --level $level --size $size --cpu_cores $cpu_cores --h5_root $h5_dir --save_root $save_dir --wsi_root $wsi_dir
#echo --datatype $datatype --wsi_format $wsi_format --level $level --size $size --cpu_cores $cpu_cores --h5_root $h5_dir --save_root $save_dir --wsi_root $wsi_dir

wsi_dir=/NAS4/llb/浙江省肿瘤医院结直肠癌数据
wsi_format="kfb;svs"
h5_dir=/NAS145/Data/LS/zz/patches_0_448/patches
save_dir=/NAS145/Data/LS/zz/image

python extract_images.py --datatype $datatype --wsi_format $wsi_format --level $level --size $size --cpu_cores $cpu_cores --h5_root $h5_dir --save_root $save_dir --wsi_root $wsi_dir
#echo --datatype $datatype --wsi_format $wsi_format --level $level --size $size --cpu_cores $cpu_cores --h5_root $h5_dir --save_root $save_dir --wsi_root $wsi_dir
#
#wsi_dir=/NAS4/llb/协和医院结直肠癌数据/slides
#wsi_format="kfb;svs;ndpi"
#h5_dir=/NAS145/Data/LS/xh/patches_0_$size/patches
#save_dir=/NAS145/Data/LS/xh/image
#
#python extract_images.py --datatype $datatype --wsi_format $wsi_format --level $level --size $size --cpu_cores $cpu_cores --h5_root $h5_dir --save_root $save_dir --wsi_root $wsi_dir