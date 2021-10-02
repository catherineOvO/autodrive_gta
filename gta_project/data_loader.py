import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


# w = [1,0,0,0,0,0,0,0,0]
# s = [0,1,0,0,0,0,0,0,0]
# a = [0,0,1,0,0,0,0,0,0]
# d = [0,0,0,1,0,0,0,0,0]
# wa = [0,0,0,0,1,0,0,0,0]
# wd = [0,0,0,0,0,1,0,0,0]
# sa = [0,0,0,0,0,0,1,0,0]
# sd = [0,0,0,0,0,0,0,1,0]
# nk = [0,0,0,0,0,0,0,0,1]

# None: 0 W:1 S:2, N:0 A:1 D:2
mapping_ = {0: (1, 0), 1: (2, 0), 2: (0, 1), 3: (0, 2), 4: (1, 1), 5: (1, 2), 6: (2, 1), 7:(2, 2), 8: (0, 0)}


valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


class LoadGTA(Dataset):
    def __init__(self, data_path, transforms_=valid_transform):
        data_tmp = []
        label_tmp = []
        self.data = []
        self.label = []
        self.transforms = transforms_

        for i in range(len(os.listdir(data_path))):
            datas = np.load(data_path + 'training_data-' + str(i + 1) + '.npy', allow_pickle=True)
            data_tmp.append(datas[:, 0])
            label_tmp.append(datas[:, 1])

        self.data = np.concatenate(np.asarray(data_tmp), axis=0)
        self.label = np.concatenate(np.asarray(label_tmp), axis=0)

        # for i in range(4, len(data_tmp)):
        #     self.data.append(torch.stack((self.transforms(data_tmp[i-4]), self.transforms(data_tmp[i-3]),
        #                     self.transforms(data_tmp[i-2]), self.transforms(data_tmp[i-1]), self.transforms(data_tmp[i]))))
        #     self.label.append(label_tmp[i])


    def __getitem__(self, item):
        tem = []
        for i in range(5):
            tem.append(self.transforms(self.data[item + i]))
        this_label = np.argmax(self.label[item + 4])
        this_img = torch.stack(tem)

        # img, (label0, label1)
        return (this_img, *mapping_[this_label])


    def __len__(self):
        return len(self.data) - 4


def data_loader(file_path, batch_size=1, num_workers=0, shuffle=False):
    params = {'batch_size': batch_size,
              'shuffle': shuffle,
              'num_workers': num_workers}

    loader = LoadGTA(file_path)
    training_generator = DataLoader(loader, **params)
    return training_generator


if __name__ == '__main__':
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}

    loader = LoadGTA('datas/data2/d2/')

    training_generator = DataLoader(loader, **params)

    co1 = np.zeros(3)
    co2 = np.zeros(3)

    for X, y0, y1 in training_generator:
        # print(X.shape)
        # print(y0, y1)
        co1[y0] += 1
        co2[y1] += 1

    print(co1)
    print(co2)
