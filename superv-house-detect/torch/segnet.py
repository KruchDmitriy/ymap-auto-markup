import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from dataLoader import spacenetLoader
from lr_scheduling import *
from torch_utils import *


class segnet(nn.Module):

    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
        super(segnet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1


    def init_vgg16_params(self, vgg16):
        blocks = [self.down1,
                  self.down2,
                  self.down3,
                  self.down4,
                  self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit]
            else:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit,
                         conv_block.conv3.cbr_unit]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    # print(input.size())

    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    # print(log_p)
    # print(log_p.size())

    target = target.view(n * h * w)
    # print(target)
    # print(target.size())

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)

    if size_average:
        loss /= mask.data.sum()
    return loss


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu


def train(args):
    # Setup Dataloader
    data_loader = spacenetLoader(args.img_rows, args.img_cols)
    trainloader = data.DataLoader(data_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # Setup visdom for visualization
    vis = visdom.Visdom()

    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    model = segnet(data_loader.n_classes)
    vgg16 = models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
    print('model initialized')

    if torch.cuda.is_available():
        model.cuda(0)
        test_image, test_segmap = data_loader[0]
        test_image = Variable(test_image.unsqueeze(0).cuda(0))
    else:
        test_image, test_segmap = data_loader[0]
        test_image = Variable(test_image.unsqueeze(0))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)

            iter = len(trainloader)*epoch + i
            poly_lr_scheduler(optimizer, args.l_rate, iter)

            optimizer.zero_grad()
            outputs = model(images)

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizer.step()

            vis.line(
                X=torch.ones((1, 1)).cpu() * iter,
                Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                win=loss_window,
                update='append')

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))

        torch.save(model, "{}_{}_{}.pkl".format("rio", "segnet_model", epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='../../preprocessedData/',
                        help='Path to dataset')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    args = parser.parse_args()
    train(args)
