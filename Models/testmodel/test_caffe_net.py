import caffe
net = caffe.Net('./cut.prototxt', './dst.caffemodel',1)
