require 'loadcaffe'

model = loadcaffe.load('VGG_ILSVRC_16_layers_deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', 'cudnn')
print(model)
