from typing import List,Tuple
import glob
import random
import os
from PIL import Image,ImageEnhance
import numpy as np
from neuthink.graph.basics import Graph,NodeList,Node
from functools import reduce
import copy


class mvImage():
    def __init__(self, path:str,in_memory=False,size=None,mode = None, source=None)->None:
        '''mvImage constructor
           Args:
               path -- path to image
               in_memory -- store image in memory
              _size -- resize image to given_size
               mode -- convert image to this mode ('L', or 'RGB')
               source -- PIL image for initializing from source image (in_memory must be True)
        '''

        self.path = path
        self._content = None
        self._imarray = None
        self._size = size
        self._in_memory = in_memory
        self._mode = mode
        
        if in_memory:
            if source is None:
                self._content  = Image.open(path)
                if size!=None:
                    self._content = self._content.resize(size, resample=Image.BICUBIC)
                if mode!=None:
                    self._content = self._content.convert(mode)


                self._imarray = np.array(self._content)
            
            
            else:
                print(type(source))
                if type(source) == np.ndarray:
                    print("hello")
                    self._content = Image.fromarray(source)
                else:
                    self._content = source
                if size!=None:

                      self._content = self._content.resize(size, resample=Image.BICUBIC)
                if mode!=None:
                    self._content = self._content.convert(mode)

                #self._imarray = np.array(self._content)
            self._size = self._content.size
            #print(self._size)

    @property
    def content(self) -> Image:
            '''returns PIL image object'''
            if self._content is not None:
                return self._content
            else:
                im = Image.open(self.path)
                if self._size!=None:
                    im = im.resize(self._size,resample=Image.BICUBIC)
                    im.save("g.png")
                if self._mode!=None:
                    im = im.convert(self._mode)
            return im
        #content = property(__get_content)

    @property
    def size(self):
        if self._in_memory:
            return self._size
        else:
            return self.content.size

    @size.setter
    def size(self, value:Tuple[int,int]):
        if self._in_memory:
            self._content = self.content.resize(value, Image.BICUBIC)
        self._size = value

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]    
    
    
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value:str):
        if self._in_memory:
            self._content = self.content.convert(value)
        self._mode = value

    @property
    def imarray(self):
            '''returns numpy array'''
            #self.content.show()
            u  =(np.array(self.content) /255.0)

            return u
           # print(z)
            #return (np.array(self.content) /255.0)
    
    def content_array_func(self, func):
        if func is not None:
            new_c = (np.array(func(self.content)))/255
        else:
            return self.imarray 
        return new_c
    
    @property
    def noisy_content(self):
        '''returns image after a set of random transformations, suitable for data augmentation'''

        img = self.content
        q = random.randint(0,360)
        #for i in range(0,q):
        #    img = img.rotate(90)
        img = img.rotate(q)
       # img = orig
        img = img.resize((224 + random.randint(0,40),224 + random.randint(0,40)),Image.BICUBIC)
        img = img.crop((0,0,224,224))
        contrast = ImageEnhance.Contrast(img)
        contrast_applied = contrast.enhance(random.random()+0.5)
        brightness = ImageEnhance.Brightness(img)
        contrast_applied=brightness.enhance(random.random()+0.5)

        z = ((np.array(img) /255.0)) #  + np.random.normal((128,128,3),0.01)
      #  print(z.shape)
        return z

    def Show(self):
        '''displays image'''
        self.content.show()
    
    def Save(self, filename, imformat='png'):
        '''saves image'''
        self.content.save(filename,imformat)
    
    def Area(self):
        return self.content.size[0] * self.content.size[1]


    def _faster_threshold(self, threshold, window_r, invert):
        from scipy import ndimage
        percentage = threshold / 100.
        window_diam = 2*window_r + 1
        image = self.content
        # convert image to numpy array of grayscale values
        img = np.array(image.convert('L')).astype(np.float) # float for mean precision 
        # matrix of local means with scipy
        means = ndimage.uniform_filter(img, window_diam)
        # result: 0 for entry less than percentage*mean, 255 otherwise 
        height, width = img.shape[:2]
        if not invert:
            result = np.zeros((height,width), np.uint8)   # initially all 0
            result[img >= percentage * means] = 255        
        else:
            result = np.ones((height,width), np.uint8) * 255   # initially all 0
            result[img >= percentage * means] = 0      
        # convert back to PIL image
        return Image.fromarray(result)

    def _slow_thresold(self, threshold, windowsize, invert):
        #this function is very slow now, need fixing
        image2 = copy.copy(self.content).convert('L')
        ws = windowsize
        w, h = self.content.size
        l = self.content.convert('L').load()
        l2 = image2.load()
        threshold /= 100.0
        if invert:
            high = 0
            low = 255
        else:
            high = 255
            low = 0
        for y in range(h):
            for x in range(w):
                #find neighboring pixels
                neighbors =[(x+x2,y+y2) for x2 in range(-ws,ws) for y2 in range(-ws, ws) if x+x2>0 and x+x2<w and y+y2>0 and y+y2<h]
                #mean of all neighboring pixels
                mean = sum([l[a,b] for a,b in neighbors])/len(neighbors)
                if l[x, y] < threshold*mean:
                    l2[x,y] = low
                else:
                    l2[x,y] = high
        return image2

    def Threshold(self, threshold:int=75, windowsize:int=10, invert=True):
        '''Returns new image converted to monochrome, using adaptive (Bradley) thresold
            Args:
             thresold - cutoff thresold
             windowsize - size of the window (in pixels) for which adaptive thresold is computed

        '''
        slow = False
        try:
            from scipy import ndimage
        except:
            print("Scipy not installed, using slow thresholding function...")
            slow = True
        if slow:
            image2 =  self._slow_thresold(threshold, windowsize, invert)
        else:
            image2 = self._faster_threshold(threshold, windowsize, invert)


        return mvImage("", in_memory=True, source=image2)

    def GetConnectedObjects(self, minsize=5):
       '''computes bounding boxes of all connected components of an imageself.
       this function requires scipy and imports it locally to avoid whole module to depend on it
       RETURNS
         List of Tuple[(top_left,bottom_right) bounding box coords * cropped mvImage]
       '''
       arr = np.array(self.content)
       from scipy import ndimage       #yes, this is not good, but we want this behavior

       s = [[1,1,1],[1,1,1],[1,1,1]]
       labeled, nr_objects = ndimage.label(arr > 170, structure=s)
       print(nr_objects)
       objects = ndimage.measurements.find_objects(labeled)
       p = []
       for x in objects:
           point1 = (x[1].start, x[0].start)
           point2 = (x[1].stop,  x[0].stop)
           im = mvImage("",in_memory=True,source=self.content.crop((x[1].start, x[0].start,x[1].stop,  x[0].stop)))
           if im.content.size[0] * im.content.size[1] > minsize:
              p.append(((point1,point2),im))
       return p




   #     self.model.metadic[name] = NodeList.from_list(image_list, self.graph)
   #     self.model.capactity = len(self.model.metadic)
   #     print("Loaded " + str(len(image_list)) + " images")
   #     res = image_list
   #     self.model.class_source = name
   #     self.model.last_call = name
   #     self.model.mtype ="Normal"
   #     self.model.start_point = name
   #     self.model.record('Image.Load',['name','input'],[name,input])
