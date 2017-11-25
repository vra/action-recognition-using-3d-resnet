if [ "$#" -ne 1 ]; then
	echo "usage: $0 dataset" >&2
	exit 1
fi

RES3D_PATH=/data1/yunfeng/Lab/video-classification-3d-cnn-pytorch
DATASET=$1
CUDA_VISIBLE_DEVICES=0 python3 $RES3D_PATH/main.py --input data/lists/$DATASET/video_list.txt --video_root /data1/yunfeng/dataset/$DATASET/videos --output data/jsons/$DATASET/resnet_3d_34.json --model data/models/resnet-34-kinetics.pth --mode feature 

