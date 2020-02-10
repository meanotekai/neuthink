from typing import List,Tuple
import glob
import random
from functools import reduce
from PIL import Image
import numpy as np
from neuthink.nImage  import mvImage
from neuthink.graph.basics import Graph,NodeList,Node
import torch
import torch.nn as nn
import math

class MetaImage(object):
    """Image class, as part of metamodel"""
    def __init__(self, ParentModel):
        super(MetaImage, self).__init__()
        self.model = ParentModel
        self.graph = ParentModel.parent_graph

    def precision(self, name):
        pname ="target_class#p"
        #predicted value = class_name
        PP = self.model.metadic[name].Match({"type": "image", pname: class_name})
        TP = PP.Match({self.aspect_name: self.class_name})
        if PP.Count() > 0:
            return (TP.Count() / PP.Count())
        else:
            return 0.0

    def accuracy(self, name):
        count = 0.0
        total = 0.0
        for image in self.model.metadic[name]:
            if "target_class#p"  in image:
                    if image["target_class#p"] == image["target_class"]:
                        count = count + 1
            total = total + 1
        return count / total



    def Load(self, path:str='data', patterns:List[str]=['jpg','jpeg','png'],name="images", input=None, in_memory=False):
        '''load data from folders, where each folder represents separate class'''
        global res
        if self.model.mode == 'train':
            return

        if self.model.mode == "design":
         if input is None:
             input = name
         folders = glob.glob(path + '/*/')
         #print(folders)
         image_names = []
         image_names = reduce(lambda x,y:x+y,[[(glob.glob(x + '*.' + _type),x) for x in folders] for _type in patterns] )
         image_list : NodeList = NodeList(self.graph)
         for iname in image_names:
             images, image_class = iname
             image_class = image_class.split('/')[-2]
             image_list = image_list + [Node(self.graph,{"image":mvImage(x,in_memory = in_memory),"target_class":image_class.lower()}) for x in images]
         random.shuffle(image_list)
         self.model.metadic[name] = NodeList.from_list(image_list, self.graph)
         self.model.capactity = len(self.model.metadic)
         print("Loaded " + str(len(image_list)) + " images")
         res = image_list
         self.model.class_source = name
         self.model.last_call = name
         self.model.mtype ="Normal"
         self.model.start_point = name
         self.model.record('Image.Load',['name','input'],[name,input])

        if self.model.mode == "predict":
           if type(self.model.metadic[input]) is str:
               path = self.model.metadic[input]
               images = (glob.glob(path))
               image_list = [Node(self.graph,{"image":mvImage(x,in_memory = in_memory),"target_class":"unknown"}) for x in images]
               self.model.metadic[name] = NodeList.from_list(image_list, self.graph)
               self.model.capactity = len(self.model.metadic)


        return self

    def Resize(self, scale:Tuple[int,int]=(214,214),target=None, source=None):
        ''' Resize operation over image '''
        if source is None:
            source = self.model.last_call
        if target is None:
            target = source
        if source == target:
           for x in self.model:
               x[source].size = scale
        else:
            print("Error: Operation not implemented")
            return self

        #res =  self.model
        self.model.last_call = target
        self.model.record("Image.Resize",['scale','source','target'],[scale, source, target])
        return self

    def Mode(self, mode='L', target=None, source=None):
        '''Mode (color depth) change operation over image'''
        if source is None:
            source = self.model.last_call
        if target is None:
            target = source
        #print(source,target)
        if source == target:
            for x in self.model:
                x[source].mode = mode
        else:
            print("Error: Operation not implemented with different source and target")
            return self

        self.model.record("Image.Mode",['mode','source','target'],[mode, source, target])
        self.model.last_call = target
        return self

    def UnVectorize(self, source=None, target=None):
        if self.model.mode=="train":
            return
        if source in self.model.metatensor:
            data = self.model.metatensor[source]

            if self.model.device is not None:
                data= data.cpu()
            
            data = data.detach()
            if len(data.shape)==4:
                data = data.squeeze(1)
            data = data.numpy()
            minrange = min(self.model.batch_size, len(self.model)-self.model.index)
            for i in range(self.model.index, self.model.index+minrange):
                self.model[i][target]=mvImage("",source=Image.fromarray((data[i-self.model.index]*255).astype('float32')),in_memory=True)
            self.res = [x[target] for x in self.model[self.model.index: (self.model.index+minrange)]]
            self.model.result = self.res
        if self.model.mode=='design':
            self.model.record("Image.UnVectorize",['source','target'],[source,target])
        return self.model


    def Vectorize(self,  target=None, source=None,channel_id=-1, precision="normal", prefunc="None"):
        '''Vectorizes image '''
        if source is None:
            source = self.model.last_call
        if target is None:
            target = "image_tensor"
        

        
        if self.model.mode=='design':
            #if mode is design, limit number of elements to process
            print("INFO: Design mode, vectorizing 50 elements to test")
            maxnodes = min(50,len(self.model))
            if prefunc is not None:
                self.prefunc = prefunc
            else:
                self.prefunc = None
            nodes = self.model[0:maxnodes]
            self.model.record("Image.Vectorize",['source','target','channel_id','precision','prefunc'],[source,target,channel_id, precision,str(prefunc)])
#        print("cnahhe",channel_id)

        if len(self.model)>0 and source in self.model[0]:
            if self.model.mode=='train':
                nodes = self.model[self.model.index:self.model.index+self.model.batch_size]
            if self.model.mode=='predict' or self.model.mode=='eval':
            #nodes = self.model
                nodes = self.model[self.model.index:self.model.index+self.model.batch_size]

            if len(self.model)>0:

            z = []

            for x in nodes:
                if self.prefunc is not None and prefunc!="None" and self.model.mode=='train':
                  z.append(torch.from_numpy(x[source].imarray).float())
                else:
                  z.append(torch.from_numpy(x[source].content_array_func(self.prefunc)).float())
                             
            if channel_id != -1:
                for x in z:
                    x[target] = x['target'][:,:,channel_id]
        

#            res = [x[target] for x in nodes]
            res = z
            self.model.metatensor[target] = torch.stack(res,dim=0)
            if self.model.device is not None:
              self.model.metatensor[target] = self.model.metatensor[target].to(self.model.device)
#            print(precision)
            if precision!="normal":
#                print("HALFPREC")
                self.model.metatensor[target] = self.model.metatensor[target].half()

            self.model.res = self.model.metatensor[target]

            
            self.model.last_call = target
        return self

    ##mass processing functions, that are not recorded/tracked
    def MakeSubimages(self, size:int, stride:int, source='image'):
        ''' makes new nodelist that consists of subimages of all image'''
        import neuthink.metagraph as m # can't import before, will get circular import
        if source is None:
            source = self.model.last_call
        newnodes = m.dNodeList(self.model.parent_graph)
        for x in self.model:
            subimages = MakeSubimages(x[source], stride, size)
            
            for im in subimages:
                imnode = Node(self.model.parent_graph,{'type':'subimage','image':im})
                newnodes.append(imnode)
                x.Connect(imnode)
        return newnodes





def LoadPDF(path, in_memory=False):
    ''' this is special function for loading images from pdf file
        it actuall creates png images of pages on disk! requires WAND and ImageMagik'''
    try:
      from wand.image import Image  #another local import to avoid proliferation of auxiliary dependencies 
    except:
        print("ERROR:Can not import Wand library, that is needed for reading PDF. Please make sure wand http://docs.wand-py.org/ is installed")
        return
    import neuthink.metagraph as m # can't import before, will get circular import
    
    diag=path
    graph = Graph()
    image_list : m.dNodeList = m.dNodeList(graph)
    #if you see not authorized error, run this from console: sudo sed -i '/PDF/s/none/read|write/' /etc/ImageMagick-6/policy.xml
    #also see here https://stackoverflow.com/questions/42928765/convertnot-authorized-aaaa-error-constitute-c-readimage-453 
    with(Image(filename=diag,resolution=300)) as source:
        images=source.sequence
        pages=len(images)
        for i in range(pages):
            Image(images[i]).save(filename=str(i)+'.png')
            image = mvImage(path=str(i)+'.png', in_memory=in_memory)
            imnode = Node(graph,{'image':image,'type':'image'})
            image_list.append(imnode)
    return image_list
    

def Save(data, prefix:str, names=None, imformat:str='png'):
    for i,x in enumerate(data):
        name = x[names] if names is not None else prefix+str(i)+'.'+imformat
        x['image'].Save(name, imformat=imformat)



def Load(path:str, in_memory = False, patterns = ['png','jpg','jpeg'], target_class='target_class'):
        import neuthink.metagraph as m # can't import before, will get circular import
        def make_list(image_names, image_list, patterns):

            for iname in image_names:
                print(iname)
                images, image_class = iname
                if len(image_class)>0:
                   
                   image_class = image_class.split('/')[-2]
                #image_name = image_class.
                image_list = image_list + [Node(graph,{"image":mvImage(x,in_memory = in_memory),target_class:image_class.lower(),'type':'image'}) for x in images]
            return image_list

        folders = glob.glob(path + '/*/')
        graph = Graph()
      #  print(folders)
        if len(folders)>0:
            image_names = reduce(lambda x,y:x+y,[[(glob.glob(x + '*.' + _type),x) for x in folders] for _type in patterns] )
           # print(image_names)
            image_list : m.dNodeList = m.dNodeList(graph)
            image_list = make_list(image_names, image_list, patterns)
            random.shuffle(image_list)
        else:
        #    print("here")
            image_names = reduce(lambda x,y:x+y,[[(glob.glob(path + '/*.' + _type),"")] for _type in patterns])
            image_list : m.dNodeList = m.dNodeList(graph)
            image_list = make_list(image_names, image_list, patterns)
        image_list.last_call = "image"
        return image_list


def JoinBoundingBoxes(scene:NodeList, min_distance=None, comp_class=None):
    '''joins all components of type comp_class that are closer then mindistance or if they bounding boxes intersect '''
    #TODO: Code really needs cleanup and simplification
    #TODO: Need to have an ability to specify vertical and horizontal distance separtely
    def center(component1):
        cx = (component1[0][0] + component1[1][0])/2
        cy = (component1[0][1] + component1[1][1])/2
        return (cx,cy)
    def distance(component1, component2):
        c1 = center(component1)
        c2 = center(component2)
        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    def boundary_distance(x,y):
        left_right = sorted([x,y], key=lambda x: x[0][0])
        top_down = sorted([x,y], key=lambda x: x[1][1])
        left_d = left_right[1][0][0] - left_right[0][1][0]
        top_d = top_down[1][1][1] - top_down[0][1][1]
        return (left_d, top_d)

    is_finished = False
    while not is_finished:
        is_finished = True
        newnodes = NodeList(scene.parent_graph)
        for x in scene:
            if not "_state_delete" in x:
                for y in scene:
                    if x!=y and not "_state_delete" in y:
                        left_d,top_d = boundary_distance(x['location'],y['location'])
                        print(left_d, top_d)
                        if (left_d < min_distance and top_d < min_distance) and x['target_class']==comp_class and y['target_class']==comp_class and not "_state_delete" in x and not "_state_delete" in y:
                            p1 = x.Parents({})
                            p2 = y.Parents({})
                            if len(p1)>0 and len(p2)>0:
                              if p1[0] != p2[0]:
                                continue
                            left_right = sorted([x,y], key=lambda x: x['location'][0][0])
                            top_down = sorted([x,y], key=lambda x: x['location'][0][1])
                            print("---------------")
                            print(x,y)
                            print(left_right, top_down)
                            newbox = (left_right[0]['location'][0][0], top_down[0]['location'][0][1]),((left_right[1]['location'][1][0], top_down[1]['location'][1][1]))
                            print(newbox)
                            print("---------------")
                            new_image = mvImage("",in_memory=True,source=x['base_image']['image'].content.crop((newbox[0][0],newbox[0][1],newbox[1][0],newbox[1][1])))
                            new_image_node = Node(scene.parent_graph, {'image':new_image,'location':newbox,'base_image':x['base_image'], 'target_class':comp_class,'type':'image'})
                            newnodes.append(new_image_node)
                            x.Parents({}).First().Connect(new_image_node)
                            x['_state_delete']='yes'
                            y['_state_delete']='yes'
                            is_finished = False
        scene = scene + newnodes
    #remove
    scene.Match({"_state_delete":'yes'}).Delete()
    return


def BasicSceneGraphParser(image_node:Node, components:List[Tuple[Tuple[int,int],Tuple[int,int],mvImage]]) -> Node:
    '''this function constructs hierarchical scene graph from list of scene components with bounding boxes
    Returns:
     root node of constructed scene  graph
    '''
    def is_inside(component1, component2):
        in_x = component1[0][0] > component2[0][0] and component1[1][0] < component2[1][0]
        in_y = component1[0][1] > component2[0][1] and component1[1][1] < component2[1][1]
        return in_x and in_y
    def area(component):
       return (component[0][0] - component[1][0]) * (component[0][1] - component[1][1])
    root_node = Node(image_node.parent_graph,{type:'root','location':((0,0),(800,600))})
    #make all nodes from components
    nodes = [Node(root_node.parent_graph, {'location':c[0],'image':c[1],'base_image':image_node,'type':'image'}) for c in components]
    no_op = False
    for x in nodes:
            possible_parents = [(area(c['location']),c) for c in nodes if is_inside(x['location'],c['location'])]
            if len(possible_parents)>0:
             #we need to select only immidiate parent
             parent = min(possible_parents, key = lambda t: t[0])[1]
             #connect
             parent.Connect(x)
             no_op = False
    #connect all nodes without parents to root node
    for x in nodes:
        if len(x.Parents({}))==0:
            root_node.Connect(x)
    image_node.Connect(root_node)
    return root_node
    

def MakeSubimages(image:mvImage,stride:int, size:int)->List[mvImage]:
    '''takes one big image and splits it into subimages of given size using specified stride'''
    newimages=[]
   # print(image.width)
    for i in range(0,int(image.width),stride):
        for j in range(0,int(image.height),stride):
            
            im2 = mvImage("",in_memory=True,source=image.content.crop((i, j, i+size,  j+size)))
            newimages.append(im2)
    return newimages

    


