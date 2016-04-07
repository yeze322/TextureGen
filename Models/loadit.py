import caffe
net = caffe.Net('VGG_ave_pool_deploy.prototxt','VGG_normalised.caffemodel', 1)
