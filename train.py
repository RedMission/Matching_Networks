import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import tqdm
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloder
import matchnet


def run_epoch(total_train_batches, datatset, name='train'):
    """
    Run the training epoch
    :param total_train_batches: Number of batches to train on
    :return:
    """
    total_c_loss = 0.0
    total_accuracy = 0.0
    for i in range(int(total_train_batches)):
        x_support_set, y_support_set, x_target, y_target = dataloder.get_batch(datatset,name,batch_size=16,n_way=5,k_shot=2)
        x_support_set = Variable(torch.from_numpy(x_support_set)).float()
        y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
        x_target = Variable(torch.from_numpy(x_target)).float()
        y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

        # convert to one hot encoding
        y_support_set = y_support_set.unsqueeze(2) # 第二个维度上添加一个维度
        sequence_length = y_support_set.size()[1]
        batch_size = y_support_set.size()[0]
        y_support_set_one_hot = Variable(
            torch.zeros(batch_size, sequence_length, n_way).scatter_(2,y_support_set.data,1), requires_grad=False) # scatter_原位操作

        # reshape channels and change order
        size = x_support_set.size()
        x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
        x_target = x_target.permute(0, 3, 1, 2)
        # 输入网络计算精度和损失
        acc, c_loss = matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)
        if name == 'train':
            # optimize process
            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step()

        iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss, acc)
        print(iter_out)
        total_c_loss += c_loss
        total_accuracy += acc

    total_c_loss = total_c_loss / total_train_batches
    total_accuracy = total_accuracy / total_train_batches
    return total_c_loss, total_accuracy


if __name__ == '__main__':
    keep_prob = 0.0
    batch_size = 20
    num_channels = 1
    lr = 1e-3
    fce = True
    n_way = 20
    k_shot = 1
    image_size = 150

    optim = "adam"
    wd = 0
    # 加载数据
    x = np.load('F:\jupyter_notebook\DAGAN\datasets\IITDdata_left.npy')  # Load Data
    # print(x.shape) #(230, 6, 150, 150, 1)
    np.random.shuffle(x)  # shuffle dataset
    x_train, x_val = x[:160], x[160:]  # divide dataset in to train, val
    # Normalize Dataset
    x_train = dataloder.processes_batch(x_train, np.mean(x_train), np.std(x_train))
    x_val = dataloder.processes_batch(x_val, np.mean(x_val), np.std(x_val))
    # Defining dictionary of dataset
    datatset = {"train": x_train, "val": x_val}

    # 建立网络对象
    matchNet = matchnet.MatchingNetwork(keep_prob, batch_size, num_channels, lr, fce, n_way,
                                        k_shot, image_size)
    # 设置参数优化器
    optimizer = torch.optim.Adam(matchNet.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # Training setup
    total_epochs = 20
    total_train_batches = 100
    total_val_batches = 10
    total_test_batches = 10

    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    test_loss, test_accuracy = [], []
    for e in range(total_epochs):
        ############################### Training Step ##########################################
        total_c_loss, total_accuracy = run_epoch(total_train_batches, datatset,'train')
        train_loss.append(total_c_loss)
        train_accuracy.append(total_accuracy)

        ################################# Validation Step #######################################
        total_val_c_loss, total_val_accuracy = run_epoch(total_val_batches,datatset, 'val')
        val_loss.append(total_val_c_loss)
        val_accuracy.append(total_val_accuracy)
        print("Epoch {}: train_loss:{:.2f} train_accuracy:{:.2f} valid_loss:{:.2f} valid_accuracy:{:.2f}".
              format(e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy))

