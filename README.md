# YOLO_tiny

This implementation of [YOLO](https://arxiv.org/pdf/1506.02640.pdf) with TensorFlow.

## Installation
1. Clone YOLO_tiny repository
	```Shell
	$ git clone https://github.com/leeyoshinari/YOLO_tiny.git
    $ cd YOLO_tiny
	```

2. Download Pascal VOC2007 dataset, and put the dataset into `data/Pascal_voc`.

   if you download other dataset, you also need to modify file paths.

3. Download [yolo_tiny](https://drive.google.com/file/d/0B-yiAeTLLamRekxqVE01Yi1RRlk/view?usp=sharing), and put weight file into `data/output`.

   Or you can also download my training weights file [YOLO_tiny](https://pan.baidu.com/s/1Xf-YEAHj2PJ35ImDR-Tthw).

4. Modify configuration into `yolo/config.py`.

5. Training
	```Shell
	$ python train.py
	```

6. Test
	```Shell
	$ python test.py
	```

## Training on Your Own Dataset
To train the model on your own dataset, you should need to modefy:

1. Put all the images into the `Images` folder, put all the labels into the `Labels` folder. Select a part of the image for training, write this part of the image filename into `train.txt`, the remaining part of the image filename written in `test.txt`. Then put the `Images`, `Labels`, `train.txt` and `test.txt` into `data/dataset`. Put weight file in `data/output`.

2. `config.py` modify the CLASSES.

3. `train.py` replace`from utils.pascal_voc import pascal_voc` with `from utils.preprocess import preprocess`, and replace `pascal = pascal_voc()` with `pascal = preprocess()`.

## Requirements
1. Tensorflow
2. OpenCV
