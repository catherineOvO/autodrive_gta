import numpy as np
import torch.nn as nn
import models
import torch
import os


def adjust_input_image_size_for_proper_feature_alignment(input_img_batch, output_stride=8):

    input_spatial_dims = np.asarray(input_img_batch.shape[2:], dtype=np.float)

    new_spatial_dims = np.ceil(input_spatial_dims / output_stride).astype(np.int) * output_stride + 1

    new_spatial_dims = list(new_spatial_dims)

    input_img_batch_new_size = nn.functional.upsample_bilinear(input=input_img_batch,
                                                               size=new_spatial_dims)

    return input_img_batch_new_size


class Resnet18_8s(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet18_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = models.resnet18(fully_conv=True,
                                    pretrained=True,
                                    output_stride=8,
                                    remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)

        self.resnet18_8s = resnet18_8s

        self._normal_initialization(self.resnet18_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]

        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)

        x = self.resnet18_8s(x)

        x = nn.functional.upsample(x, size=input_spatial_dim, mode='bilinear', align_corners=True)

        # x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)#, align_corners=False)

        return x

    def forward_unfc(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]

        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)

        x = self.resnet18_8s(x, if_return_unfc=True)

        x = nn.functional.upsample(x, size=input_spatial_dim, mode='bilinear', align_corners=True)

        # x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)#, align_corners=False)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, batch_size=1, num_layers=1, drop_prob=0.5):
        # input_size: Corresponds to the number of features in the input.
        # hidden_layer_size: Specifies the number of hidden layers along with the number of neurons in each layer.
        # output_size: The number of items in the output
        super(LSTM, self).__init__()

        self.num_layers = num_layers

        self.batch_size = batch_size

        self.hidden_layer_size = hidden_layer_size
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)

        self.dropout = nn.Dropout(drop_prob)

        self.linear = nn.Sequential(nn.Linear(hidden_layer_size, 100), nn.ReLU(), nn.Dropout(drop_prob),
                                    nn.Linear(100, 100), nn.ReLU())

        self.hidden_cell = None  # self.init_hidden()

        self.out0 = nn.Linear(100, 3)
        self.out1 = nn.Linear(100, 3)

        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                     torch.zeros(1, 1, self.hidden_layer_size))

    # def init_hidden(self):
    #     if self.batch_size == 1:
    #         bz = 1
    #     else:
    #         bz = self.batch_size
    #
    #     return (torch.zeros(self.num_layers, bz, self.hidden_layer_size).cuda().requires_grad_(),
    #             torch.zeros(self.num_layers, bz, self.hidden_layer_size).cuda().requires_grad_())

    def forward(self, input_img):
        # lstm_out: [input_size, batch_size, hidden_layer_size]
        # self.hidden: (a, b), a and b are both [num_layers, batch_size, hidden_layer_size]
        lstm_out, self.hidden_cell = self.lstm(input_img, self.hidden_cell)
        self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())
        predictions = self.linear(lstm_out[-1].contiguous().view(input_img.shape[1], -1))
        return self.out0(predictions), self.out1(predictions)


class End2End(nn.Module):
    def __init__(self, input_size=336608, hidden_layer_size=100, batch_size=1, num_layers=1,
                 saved_model_path='resnet_18_8s_cityscapes_best.pth'):
        super(End2End, self).__init__()

        self.resnet = Resnet18_8s(19)
        # with torch.set_grad_enabled(False):
        #     self.resnet = Resnet18_8s(19)

        self.conv = nn.Conv2d(512, 32, kernel_size=(2, 2), stride=(2, 2))

        self.lstm = LSTM(input_size, hidden_layer_size, batch_size, num_layers)

        self.resnet.load_state_dict(torch.load(saved_model_path))

    def forward(self, x):
        x = self.resnet.forward_unfc(x)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = nn.functional.relu(x, inplace=True)
        # x = torch.flatten(x, 1)
        x = x.contiguous().view(5, 1, -1)
        # print(x.shape)
        x = self.lstm(x)
        # print(x[0].shape, x[1].shape)
        return x


if __name__ == '__main__':
    net = End2End()
    tem_ = torch.rand((5,3,270,630))
    print(net(tem_))