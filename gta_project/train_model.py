import torch
import gta_model
from data_loader import *
import os


# set params
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

num_layers = 1
max_epochs = 15
learning_rate = 1e-2

bias0 = 150
bias1 = 50

upgrad_step = 50 # super batchsize
save_direct = 'new_gta_model_bias_150_50_up50'

# train a network
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")

training_generator_1 = data_loader(file_path='datas/data2/d1/')
training_generator_2 = data_loader(file_path='datas/data2/d2/')

gta_net = gta_model.End2End(num_layers=num_layers)
gta_net = torch.nn.DataParallel(gta_net)
gta_net.cuda()
gta_net.train()

update_step = upgrad_step

# balance data
w = np.array([[1617, 10271], [9279,  600], [596,  600]])
# w = (w - np.min(w, axis=0)) / (np.max(w, axis=0) - np.min(w, axis=0))
# w = torch.from_numpy(1 - w).type('torch.FloatTensor').cuda()
w[:, 0] += bias0
w[:, 1] += bias1
w = 1 / w
w = w / np.sum(w, axis=0)
w = torch.from_numpy(w).type('torch.FloatTensor').cuda()

loss_fn1 = torch.nn.CrossEntropyLoss(weight=w[:, 0], size_average=False)
loss_fn2 = torch.nn.CrossEntropyLoss(weight=w[:, 1], size_average=False)

for epoch in range(max_epochs):
    optimizer = torch.optim.SGD(gta_net.parameters(), lr=learning_rate, momentum=0.5, weight_decay=1e-4)
    optimizer.zero_grad()
    print('====================== Epoch: %d =====================' % epoch)
    loss_epoch = 0
    counter = 1
    train_generator = training_generator_2 if epoch // 2 == 0 else training_generator_1

    tem_0 = []
    tem_1 = []

    for X, y0, y1 in train_generator:
        X = X.squeeze(0).cuda()
        # print(X.shape)
        X = torch.nn.functional.interpolate(X, scale_factor=(0.5, 0.5), mode='bilinear')

        y0 = torch.tensor(y0[::2]).cuda()
        y1 = torch.tensor(y1[::2]).cuda()
        #try:
        y_pred = gta_net(X)
        loss = [0, 0]
        loss[0] = loss_fn1(y_pred[0], y0)
        loss[1] = loss_fn2(y_pred[1], y1)
        loss_ = loss[0] + loss[1]
        loss_epoch = loss_epoch + loss_

        if counter % 100 == 0:
            print('Loss1: %.3f Loss2: %.3f' % (loss[0], loss[1]))
        counter = counter + 1

        if counter == 80:
            print(y_pred)

        loss_.backward()

        if counter % update_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        tem_0.append(torch.argmax(y_pred[0]).item())
        tem_1.append(torch.argmax(y_pred[1]).item())

        # except Exception as inst:
        #     print('Error at', epoch, X.shape)
        #     print(inst)
    tem_0 = np.array(tem_0)
    tem_1 = np.array(tem_1)
    print(np.sum(tem_0 == 0), np.sum(tem_0 == 1), np.sum(tem_0 == 2))
    print(np.sum(tem_1 == 0), np.sum(tem_1 == 1), np.sum(tem_1 == 2))
    print('Epoch %d: loss: %.5f' % (epoch, loss_epoch/counter))
    if (epoch + 1) % 6 == 0 and epoch != 0:
        learning_rate = learning_rate * 0.1
        os.makedirs(save_direct, exist_ok=True)
        torch.save(gta_net.state_dict(), os.path.join(save_direct, 'gta_model_%d.pth' % (epoch + 1)))

torch.save(gta_net.state_dict(), os.path.join(save_direct, 'gta_model.pth'))

