# SRDNet
Hyperspectral Image Super-Resolution via Dual-domain Network Based on Hybrid Convolution
This repository is implementation of the ["Hyperspectral Image Super-Resolution via Dual-domain Network Based on Hybrid Convolution"](SRDNet)by PyTorch.

Dataset
------
**Three public datasets, i.e., 
[CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ 
"CAVE"), [Harvard](http://vision.seas.harvard.edu/hyperspec/explore.html 
"Harvard"), [Chikusei](https://naotoyokoya.Com/Download.html), are employed to verify the effectiveness of the  proposed SRDNet. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, we also provide the code about data pre-processing in folder [data pre-processing](https://github.com/qianngli/MCNet/tree/master/data_pre-processing "data pre-processing"). The folder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**

Requirement
**python 3.7, Pytorch 1.7.1, cuda 11.0, RTX 3090 GPU**
Training
--------
**The ADAM optimizer with beta_1 = 0.9, beta _2 = 0.999 is employed to train our network.  The learning rate is initialized as 11^-4 for all layers, which decreases by a half at every 35 epochs.**

**You can train or test directly from the command line as such:**

###### # python train.py --cuda --datasetName CAVE  --upscale_factor 4
###### # python test.py --cuda --model_name checkpoint/model_4_epoch_XXX.pth

Result
--------
**To qualitatively measure the proposed SRDNet, three evaluation methods are employed to verify the effectiveness of the algorithm, including  Peak Signal-to-Noise Ratio (PSNR), Structural SIMilarity (SSIM), and Spectral Angle Mapper (SAM).**


| Scale  |  CAVE |  Harvard |  
| :------------: | :------------: | :------------: | :------------: | 
|  x2 |  46.124 / 0.9929 / 2.264 | 47.105/ 0.9905 / 2.218| 
|  x3 |  42.944/ 0.9869/ 2.672|  43.951 / 0.9846 / 2.443/
|  x4 | 40.412/ 0.9807 / 3.025 |  42.902 / 0.9813 / 2.756 /

| Scale  |Chikusei|  
| :------------: | :------------: | :------------: | :------------: | 
|  x4 |  37.982/ 0.981 / 2.659 | 
|  x8 |  33.854/ 0.950/ 4.435|  

If you has any questions, please send e-mail to lttdouble@163.com.
