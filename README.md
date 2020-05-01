# Source codes for paper ["SAL:Selection and Attention Losses for Weakly Supervised Semantic Segmentation"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9082835), accepted by Transaction on Multimedia

## 1. Training:
run SAL_Net_VGG16_training.py

## 2. Testing:

Step 1: download the compressed model from [Google Driver model](https://drive.google.com/file/d/1F_HcZKZmVPOXwEzGTZkmUV9GzCM4m5OW/view). Put it in the folder "./model" and unzip it. We have release the model corresponding to steps P5 presented in TABLE VI in the sumbitted manuscript. mIoU of 58.4 can be achieved for the single model.

Step 2: Run SAL_Net_VGG16_mstest.py for SAL-Net-VGG16 evaluation, the predictions with multiscale fusion will be saved in SAVE_DIR = './result/'. Mean IoU of 59.0 can be achieved on PASCAL VOC 2012 validation dataset.

Step 3: Run SAL_Net_VGG16_mscrftest.py for SAL-Net-VGG16 with multiscale fusion and CRF. 
Thre results will be saved in './result/'. Mean IoU of 61.3 can be achieved on PASCAL VOC 2012 validation dataset.

Step 4: we have provided the matlab code for evaluation. You can evaluate the resutls and obtain Iou youself. 
Please refer to https://github.com/zmbhou/IoUeval.

## 3. source codes for mask scoring are coming soon.
