# configuration
cd ../../

export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
wsi_dir=/NAS145/liaolinbo/Data/MXB/补做的片子
wsi_format=svs
coord=/NAS145/liaolinbo/Data/MXB/补做的片子/patch_0_996
h5_dir=$coord/patches
save_dir=/NAS145/liaolinbo/Data/MXB/补做的片子/images
datatype="auto"
level=0
size=996
cpu_cores=50

python extract_images.py --datatype $datatype --wsi_format $wsi_format --level $level --size $size --cpu_cores $cpu_cores --h5_root $h5_dir --save_root $save_dir --wsi_root $wsi_dir
