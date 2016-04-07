import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Misc import *

IF_MASK = False
lossList = []

def getLossList():
    return lossList

def ImageSyn(net, constraints, init=None, bounds=None, callback=None, minimize_options=None, gradient_free_region=None):
    global lossList
    lossList = []
    resetI()
    '''
    This function generates the image by performing gradient descent on the pixels to match the constraints.

    :param net: caffe.Classifier object that defines the network used to generate the image
    :param constraints: dictionary object that contains the constraints on each layer used for the image generation
    :param init: the initial image to start the gradient descent from. Defaults to gaussian white noise
    :param bounds: the optimisation bounds passed to the optimiser
    :param callback: the callback function passed to the optimiser
    :param minimize_options: the options passed to the optimiser
    :param gradient_free_region: a binary mask that defines all pixels that should be ignored in the in the gradient descent   
    :return: result object from the L-BFGS optimisation
    '''
    if init == None:
        init = np.random.randn(*net.blobs['data'].data.shape) # '*' -> tuple to multi args
        #init = np.zeros(net.blobs['data'].data.shape)

     #get indices for gradient
    layers, indices = get_indices(net, constraints)
    
    #function to minimise 
    def f(x):
        x = x.reshape(*net.blobs['data'].data.shape)
        lastChoosenLayer = layers[min(len(layers)-1, indices[0]+1)]
        net.forward(data=x, end=lastChoosenLayer)
        f_val = 0
        #clear gradient in all layers
        for index in indices:
            net.blobs[layers[index]].diff[...] = np.zeros_like(net.blobs[layers[index]].diff)
                
        for i,index in enumerate(indices):
            layer = layers[index]
            #only one loss function
            for l,loss_function in enumerate(constraints[layer].loss_functions):
                constraints[layer].parameter_lists[l].update({'activations': net.blobs[layer].data.copy()})
                loss, grad = loss_function(**constraints[layer].parameter_lists[l])
                f_val += loss
                net.blobs[layer].diff[:] += grad
            #gradient wrt inactive units is 0
            #[???? is it needed???]
            if IF_MASK:
                net.blobs[layer].diff[(net.blobs[layer].data == 0)] = 0.

            if index == indices[-1]:
                f_grad = net.backward(start=layer)['data'].copy()
            else:        
                net.backward(start=layer, end=layers[indices[i+1]])                    

        #ignore piexels under free region
        if gradient_free_region!=None:
            f_grad[gradient_free_region==1] = 0    

        print "loss={},{}".format(f_val,IF_MASK)
        lossList.append(f_val)
        return [f_val, np.array(f_grad.ravel(), dtype=float)]            
        
    result = minimize(f, init,
                          method='L-BFGS-B', 
                          jac=True,
                          bounds=bounds,
                          callback=callback,
                          options=minimize_options)

    print "IF_MASK:{%d}"%IF_MASK
    return result

def alignPlot(fts, cols):
    fts = fts[0]
    sp = fts.shape # 50 channels
    ROWs = int(np.ceil(sp[0]/cols))
    fg = plt.figure()
    for i in range(0, ROWs):
        toshow = fts[i*cols:i*cols+cols,:,:].transpose(1,0,2)
        tsp = toshow.shape
        toshow = toshow.reshape(tsp[0],tsp[1]*tsp[2])
        fg.add_subplot(ROWs,1,i+1).imshow(toshow, cmap='gray')
    fg.show()
    return

def showFeatureMap(net, data, layer, cols=10):
    ret = net.forward(data=data, blobs=net.blobs.keys())
    fts = ret[layer]
    alignPlot(fts, cols)
    return
