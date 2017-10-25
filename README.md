# VGG16-places365 model for scene classification, written in Keras 2.0 

![Keras logo](https://i.imgur.com/c9r5WFp.png) 

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/GKalliatakis/Keras-VGG16-places365/blob/master/LICENSE)

## You have just found the Keras (2.0) implementation of the pre-trained VGG 16 convolutional-layer CNN model on Places365-Standard (~1.8 million images from 365 scene categories).


### Overview
CNNs trained on Places365 database (latest subset of [Places2 Database](http://places2.csail.mit.edu)) could be directly used for scene recognition, while the deep scene features from the higher level layer of CNN could be used as generic features for visual recognition.

### Paper
The Keras models has been obtained by directly converting the [Caffe model](https://github.com/CSAILVision/places365) provived by the authors (all the original Caffe-based resources can be found there).

More details about the network architecture can be found in the following paper:

    Places: A 10 million Image Database for Scene Recognition
    Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A.
    IEEE Transactions on Pattern Analysis and Machine Intelligence

Please consider citing the paper if you use the pre-trained CNN models.


### Contents:
Models:
- `vgg16_places_365.py`
- `vgg16_hybrid_places_1365.py`

Weights:
- [VGG16_Places365_weights.h5](https://drive.google.com/open?id=0B98ZKBhlAtp-Y2lfbHVBY3k4SjA)
- [VGG16_Hybrid_Places1365_weights.h5](https://drive.google.com/open?id=0B98ZKBhlAtp-dHNWM0x2bnM3cU0)

Serialised JSONs:
- [VGG16_Places365.json](https://drive.google.com/open?id=0B98ZKBhlAtp-Vl9GeWw0UnhQTTA)
- [VGG16_Hybrid_Places1365.json](https://drive.google.com/open?id=0B98ZKBhlAtp-c3NZdnduN2oxYUE)

Usage: Download the weights which were dumped out from the initial Caffe resources and set the argument `weights_path` accordingly in `vgg16_places_365.py` & `vgg16_hybrid_places_1365.py`

### Keras Model Summary:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
data (InputLayer)            (None, 3, 224, 224)       0         
_________________________________________________________________
conv1_1 (Conv2D)             (None, 64, 224, 224)      1792      
_________________________________________________________________
conv1_2 (Conv2D)             (None, 64, 224, 224)      36928     
_________________________________________________________________
pool1 (MaxPooling2D)         (None, 64, 112, 112)      0         
_________________________________________________________________
conv2_1 (Conv2D)             (None, 128, 112, 112)     73856     
_________________________________________________________________
conv2_2 (Conv2D)             (None, 128, 112, 112)     147584    
_________________________________________________________________
pool2 (MaxPooling2D)         (None, 128, 56, 56)       0         
_________________________________________________________________
conv3_1 (Conv2D)             (None, 256, 56, 56)       295168    
_________________________________________________________________
conv3_2 (Conv2D)             (None, 256, 56, 56)       590080    
_________________________________________________________________
conv3_3 (Conv2D)             (None, 256, 56, 56)       590080    
_________________________________________________________________
pool3 (MaxPooling2D)         (None, 256, 28, 28)       0         
_________________________________________________________________
conv4_1 (Conv2D)             (None, 512, 28, 28)       1180160   
_________________________________________________________________
conv4_2 (Conv2D)             (None, 512, 28, 28)       2359808   
_________________________________________________________________
conv4_3 (Conv2D)             (None, 512, 28, 28)       2359808   
_________________________________________________________________
pool4 (MaxPooling2D)         (None, 512, 14, 14)       0         
_________________________________________________________________
conv5_1 (Conv2D)             (None, 512, 14, 14)       2359808   
_________________________________________________________________
conv5_2 (Conv2D)             (None, 512, 14, 14)       2359808   
_________________________________________________________________
conv5_3 (Conv2D)             (None, 512, 14, 14)       2359808   
_________________________________________________________________
pool5 (MaxPooling2D)         (None, 512, 7, 7)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc6 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
drop6 (Dropout)              (None, 4096)              0         
_________________________________________________________________
fc7 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
drop7 (Dropout)              (None, 4096)              0         
_________________________________________________________________
fc8a (Dense)                 (None, 365)               1495405   
_________________________________________________________________
prob (Activation)            (None, 365)               0         
=================================================================
Total params: 135,755,949
Trainable params: 135,755,949
Non-trainable params: 0
```

### Licensing 
We are always interested in how this model is being used, so if you found this model useful or plan to make a release of code based on or using this package, it would be great to hear from you. 

### Other Models 
This is going to be an evolving repository and I will keep updating it with Keras-compatible models which are not included in [Keras Applications](https://keras.io/applications/), so make sure you have starred and forked this repository before moving on !

### Questions and Comments
If you have any suggestions or bugs to report you can pull a request or start a discussion.
_________________________________________________________________

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
