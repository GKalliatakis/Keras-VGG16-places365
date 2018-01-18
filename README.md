# Pre-trained CNN models on Places365-Standard for Keras

![Keras logo](https://i.imgur.com/c9r5WFp.png) 


[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/GKalliatakis/Keras-VGG16-places365/blob/master/LICENSE)

## You have just found the Keras models of the pre-trained CNNs on Places365-Standard (~1.8 million images from 365 scene categories).


### Overview
CNNs trained on Places365 database (latest subset of [Places2 Database](http://places2.csail.mit.edu)) could be directly used for scene recognition, while the deep scene features from the higher level layer of CNN could be used as generic features for visual recognition.

### Paper
The Keras models has been obtained by directly converting the [Caffe models](https://github.com/CSAILVision/places365) provived by the authors (all the original Caffe-based resources can be found there).

More details about the architecture of the networks can be found in the following paper:

    Places: A 10 million Image Database for Scene Recognition
    Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A.
    IEEE Transactions on Pattern Analysis and Machine Intelligence

Please consider citing the paper above if you use the pre-trained CNN models.


### Contents:
This repository contains code for the following Keras models:
- VGG16-places365
- VGG16-hybrid1365

### Usage: 
All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image dimension ordering set in your Keras configuration file at ~/.keras/keras.json. For instance, if you have set image_dim_ordering=tf, then any model loaded from this repository will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (`weights='places'` argument in model constructor for all image models). Weights are automatically downloaded.



## Examples

### Classify Places classes with VGG16-places365

```python
from vgg16_places_365 import VGG16_Places365
from keras.preprocessing import image
from places_utils import preprocess_input

model = VGG16_Places365(weights='places')

img_path = 'restaurant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predictions_to_return = 5
preds = model.predict(x)[0]
top_preds = np.argsort(preds)[::-1][0:predictions_to_return]

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

print('--SCENE CATEGORIES:')
# output the prediction
for i in range(0, 5):
    print(classes[top_preds[i]])
```

### Extract features from images with VGG16-hybrid1365

```python
from vgg16_hybrid_places_1365 import VGG16_Hubrid_1365
from keras.preprocessing import image
from places_utils import preprocess_input

model = VGG16_Hubrid_1365(weights='places', include_top=False)

img_path = 'restaurant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```


### References

- [A 10 million Image Database for Scene Recognition](http://places2.csail.mit.edu/PAMI_places.pdf) - please cite this paper if you use the VGG16-places365 or VGG16-hybrid1365 model in your work.
- [Learning Deep Features for Scene Recognition using Places Database](https://arxiv.org/abs/1512.03385)


Additionally, don't forget to cite this repo if you use these models:

    @misc{gkallia2017keras_places365,
    title={Keras-Places},
    author={Grigorios Kalliatakis},
    year={2017},
    publisher={GitHub},
    howpublished={\url{https://github.com/GKalliatakis/Keras-VGG16-places365}},
    }


### Licensing 
- All code in this repository is under the MIT license as specified by the LICENSE file.
- The VGG16-places365 and VGG16-hybrid1365 weights are ported from the ones [released by CSAILVision](https://github.com/CSAILVision/places365) under the [MIT license](https://github.com/CSAILVision/places365/blob/master/LICENSE).

We are always interested in how these models are being used, so if you found them useful or plan to make a release of code based on or using this package, it would be great to hear from you. 

### Where to get other trained models?
More info on downloading, converting, and submitting other models can be found on the main [Keras | Application Zoo repository](https://github.com/GKalliatakis/Keras-Application-Zoo).

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
