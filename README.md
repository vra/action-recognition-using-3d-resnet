# action-recognition-using-3d-resnet
Use [3D ResNet](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features of UCF101 and HMDB51 and then classify them.

# how to use

 1. Clone this repo:
 ```bash
	git clone https://github.com/vra/action-recognition-using-3d-resnet.git
 ```

 2. Download [3D ResNet](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)

 3. Download its pretrained [models](https://github.com/kenshohara/3D-ResNets-PyTorch/releases), put these models to this repo's `data/models/`

 4. run the script under `scripts` under to extract 3D resnet features of UCF101 and HMDB51:
 ```bash
	bash scripts/extract_resnet_3d_features.sh /path/to/video-classification/3d-cnn-pytorch ucf101 /path/to/ucf101/videos 
	bash scripts/extract_resnet_3d_features.sh /path/to/video-classification/3d-cnn-pytorch hmdb51 /path/to/hmdb51/videos 
 ```
Also, you can download my extracted features of ucf101 and hmdb51 at [here](https://drive.google.com/open?id=12BM8ibl5oFziM-59JqXmsqMjtx7_qthZ) and [here](https://drive.google.com/open?id=178U8N6dPBfpaHYMxdOCCWpLa4hl6kFjk). **Remember to put the first one to `data/jsons/ucf101` before you download the second one, otherwise the first one will be convered.**

5. Run `main.py` to classify extracted 3D resnet features:
 ```bash
	python main.py -dataset hmdb51
 ```
Results:

strategy | dataset | accuracy
-------- | ------- | -------
mean	 | ucf101  | 0.8487
max	     | ucf101  | 0.8667
mean	 | hmdb51  | 0.5425
max	     | hmdb51  | 0.5399



  
