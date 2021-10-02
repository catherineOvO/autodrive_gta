import os
import time
import threading

import torch
from torchvision import transforms

import numpy as np

from gta_model import End2End

from screen_operation import ScreenShoter
from directkeys import PressKey, ReleaseKey, W, A, S, D

time.sleep(10)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

num_layers = 1
t_time = 0.09

Model_Name = 'gta_model/gta_model_6.pth'

gta_model = End2End(num_layers=num_layers)
gta_model.cuda()
gta_model = torch.nn.DataParallel(gta_model)
gta_model.load_state_dict(torch.load(Model_Name))
gta_model.eval()

valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

shoter = ScreenShoter()

grab_screen = [torch.zeros((5,3,630,270)), 0]
action = [None, 0]


def car_moves(action):
    # print(action)
    # return
    if action == [0, 0]:
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(D)
        time.sleep(t_time)
    elif action == [1, 0]:
        PressKey(W)
    elif action == [2, 0]:
        ReleaseKey(W)
        PressKey(S)
    elif action == [0, 1]:
        ReleaseKey(W)
        ReleaseKey(S)
        PressKey(A)
        time.sleep(t_time)
        ReleaseKey(A)
    elif action == [0, 2]:
        ReleaseKey(W)
        ReleaseKey(S)
        PressKey(D)
        time.sleep(t_time)
        ReleaseKey(D)
    elif action == [1, 1]:
        ReleaseKey(S)
        PressKey(W)
        PressKey(A)
        time.sleep(t_time)
        ReleaseKey(A)
    elif action == [1, 2]:
        ReleaseKey(S)
        PressKey(W)
        PressKey(D)
        time.sleep(t_time)
        ReleaseKey(D)
    elif action == [2, 1]:
        ReleaseKey(W)
        PressKey(S)
        PressKey(A)
        time.sleep(t_time)
        ReleaseKey(A)
    else:
        ReleaseKey(W)
        PressKey(S)
        PressKey(D)
        time.sleep(t_time)
        ReleaseKey(D)


def screen_shot(grab_screen):
    # global grab_screen
    while True:
        # grab_screen.append([13213,1231,131])
        x = np.asarray(shoter.shot().resize((270, 630)))
        x = valid_transform(x)
        # y = np.asarray(shoter.shot().resize((630, 270)))
        # y = valid_transform(y)
        # x = x.view(1, *x.shape)
        grab_screen[0] = torch.cat((x.view(1, *x.shape), grab_screen[0][0:4]), dim=0)
        grab_screen[1] = 1


def operations(action):
    # global action
    while True:
        while action[1] == 0:
            time.sleep(1e-2)
        action[1] = 0
        car_moves(action[0])


def run_network(grab_screen, action):
    # global action
    f = open('records.txt', 'w')
    try:
        while True:
            while grab_screen[1] == 0:
                time.sleep(1e-2)
            grab_screen[1] = 0
            pred0, pred1 = gta_model(torch.nn.functional.interpolate(grab_screen[0].cuda(), scale_factor=(0.5, 0.5), mode='bilinear'))
            pred0 = pred0.detach().cpu().numpy()
            pred1 = pred1.detach().cpu().numpy()
            # pred = np.random.random((2, 5))
            f.write(str((pred0, pred1)))
            action[0] = [np.argmax(pred0), np.argmax(pred1)]
            action[1] = 1
    except:
        pass
    finally:
        f.close()

t1 = threading.Thread(target=screen_shot, args=[grab_screen]) # {'grab_screen':})
t2 = threading.Thread(target=operations, args=[action])# {'action': action})
t3 = threading.Thread(target=run_network, args=[grab_screen, action] )#{'grab_screen': grab_screen, 'action': action})
t1.start()
t2.start()
t3.start()

t1.join()

t2.join()
t3.join()