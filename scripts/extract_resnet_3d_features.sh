if [ "$#" -ne 3 ]; then
	echo "usage: $0 /path/to/3d_resnet_repo dataset /path/to/ucf01_or_hmdb51/videos" >&2
	exit 1
fi

RES3D_PATH=$1
DATASET=$2
VIDEO_ROOT=$3
CUDA_VISIBLE_DEVICES=0 python3 $RES3D_PATH/main.py --input data/lists/$DATASET/video_list.txt --video_root $VIDEO_ROOT --output data/jsons/$DATASET/resnet_3d_34.json --model data/models/resnet-34-kinetics.pth --mode feature 

