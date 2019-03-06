# Introduction
NeuThink is experimental deep learning library build on top of [pytorch](pytorch.org) Neuthink is research project aimed to explore how the concept of differential programming can be implemented in context of Python language.
One of definitions of differential programming is that it "*enables programmers to write program sketches with slots that can be filled with behavior trained from program input-output data*" ([\[Bošnjak et all, 2016\]](https://arxiv.org/abs/1605.06640) The goal of Neuthink is too see how we can express such sketches in a way that can simplify creating complex deep  learning models for non-experts.

## Motivational Example
Consider the simple classical task of classification of figures into fixed number of classes: Circles, Rectangles and Triangles ([example dataset](https://www.dropbox.com/s/7tz0r0e7hno2250/train_figures.zip?dl=0)). Lets say, we want to build a program that takes the list of figure images and selects only Triangles. The sketch of such program may look like this (in pseudo-python):
```python
list_of_figures = load_from_folder(folder_name)
filter(lambda x: x==“triangle“, map(get_shape, list_of_figures))
```
Here we specify that we need to load figures from some folder, apply some function get_shape to it and then filter only triangles. In this case the behavior that need to be learned from data get_shape() function. In ideal world we would be able to just specify dataset of known figures and tell to compiler to make this get_shape() somehow. In reality, however, there exist many details, that need to be specified, so with most popular deep learning frameworks one can not just do it - lot of other code need to written that takes care of data loading, model specification, model training  and prediction for a new data.
Neuthink aims to eliminate all that code, so examples like above could work.
Here is real Neuthink-based code that is sufficient to solve the task:

```python
import neuthink.metaimage as Image
#load data
images = Image.Load(«./train_figures»)
#specify sketch - resize image, change to grayscale and map from image to target class
images = images.Resize(scale=(32,32),source="image").Mode().dMap(target="target_class")
get_shape = images.compile()  #this actually trains the model, filling in missing behavior. this returns function get_shape

Test = Image.Load("./test") #load some test images
get_shape(Test) #the new function can be just applied to the data to get new predictions
Images.Export("get_shape") # we can export this new function for future use

# so all we will need to use it is just import
from get_shape import get_shape
```
Neuthink currently supports simple image classification models, text classification, sequence tagging and character level language models. Also, more complex models can be constructed by combining existing primitives. 


# Installation
IMPORTANT: THIS IS THE WORK IN PROGRESS AND RESEARCH PROJECT. NOT RECOMMENDED FOR PRODUCTION USE, POORLY DOCUMENTED AND CONTAINS KNOWN BUGS. USE FOR YOUR OWN RISK

No installation/setup script  is avaliable at the the time, if you want to use the library, it need to be installed manually

 - Copy neuthink folder to your python libs/dist-packages or create symlink there
 - Download pre-trained wordvectors
 - For English language:
https://www.dropbox.com/s/7xuvry8y9k85fne/vectors_en_100.txt?dl=1
https://www.dropbox.com/s/ancaj3976it1rr6/words_en_100.txt?dl=1
 - For Russian language
https://www.dropbox.com/s/64buqbpfbmvyqba/words_ru_50c.txt?dl=1
https://www.dropbox.com/s/9alxhcpq3u8jjzj/vectors_ru_50c.txt?dl=1
Note: These are not good word embeddings and are provided as example only
Copy them to ...neuthink/wordvector folder

3. Install all dependencies
 - Pystemmer (pip install pystemmer)
 - Pytorch (see https://pytorch.org/ for details )
 - Pillow pip install pillow
 - reprint (pip install reprint)


