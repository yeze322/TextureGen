import caffe
import DeepImageSynthesis.Misc as misc
import objsaver
import numpy


VGGmodel = 'Models/testmodel/cut.prototxt'
VGGweights = 'Models/testmodel/dst.caffemodel'
source_img_name = '/home/yeze/Documents/gitxivprojects/DeepTextures/Images/lishu/12.jpg'
imagenet_mean = numpy.array([ 0.40760392,  0.45795686,  0.48501961]) #mean for color channels (bgr)

misc.load_resources(source_img_name, 36., VGGmodel, VGGweights, imagenet_mean, False)

#mean = objsaver.load_obj('mean.pkl')
#net = misc.load_net('Models/testmodel/cut.prototxt','Models/testmodel/dst.caffemodel',mean)
