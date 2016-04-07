import numpy
from numpy import *
#%pylab inline
import glob
import sys
import os
from collections import OrderedDict
import caffe
from DeepImageSynthesis import *

#set caculation mode and choose gpu device
gpu = 0
caffe.set_mode_gpu() #for cpu mode do 'caffe.set_mode_cpu()'
caffe.set_device(gpu)

#Path parameters
#VGGweights = '/home/yeze/Documents/gitxivprojects/DeepTextures/Models/VGG_normalised.caffemodel'
#VGGmodel = '/home/yeze/Documents/gitxivprojects/DeepTextures/Models/VGG_ave_pool_deploy.prototxt'
VGGweights = 'Models/testmodel/dst.caffemodel'
VGGmodel = 'Models/testmodel/cut.prototxt'
source_img_name = '/home/yeze/Documents/gitxivprojects/DeepTextures/Images/lishu/12.jpg'
imagenet_mean = numpy.array([ 0.40760392,  0.45795686,  0.48501961]) #mean for color channels (bgr)

#load source image
im_size = 200.
source_img, net = load_resources(source_img_name, im_size, 
                            VGGmodel, VGGweights, imagenet_mean, 
                            show_img=False)
im_size = asarray(source_img.shape[-2:])
initpic = caffe.io.load_image('./Images/songti/129.jpg')

#l-bfgs parameters optimisation
maxiter = 2
m = 10

#define layers to include in the texture model and weights w_l
tex_layers = ['pool2', 'pool1', 'conv1']
tex_weights = [1e9,1e9,1e9]

#pass image through the network and save the constraints on each layer
constraints = OrderedDict()
net.forward(data = source_img)
for l,layer in enumerate(tex_layers):
    constraints[layer] = constraint([LossFunctions.gram_mse_loss],
                                    [{'target_gram_matrix': gram_matrix(net.blobs[layer].data),
                                      'weight': tex_weights[l]}])
#get optimisation bounds
bounds = get_bounds([source_img],im_size)

#generate new texture
result = ImageSyn(net, constraints, bounds=bounds,
                  #callback=lambda x: show_progress(x,net), 
                  minimize_options={'maxiter': maxiter,
                                    'maxcor': m,
                                    'ftol': 0, 'gtol': 0})
