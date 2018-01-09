import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segnet import *
import matplotlib.pyplot as plt
from dataLoader import spacenetLoader
from torch.autograd import Variable

loader = spacenetLoader(256, 256)
img, lab = loader.get_test_item()

c, h, w = img.size()
img = img.view([1, c, h, w])
# print(img.size())
# input()

model = torch.load('rio_segnet_model_99.pkl')
model.eval()

if torch.cuda.is_available():
    model.cuda(0)
    img = Variable(img.cuda(0))
else:
    img = Variable(img)

output = model(img)
pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=1)
pred[pred[0] == 1] = 255

plt.imshow(pred)
plt.show()
