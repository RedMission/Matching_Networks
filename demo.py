import argparse

import numpy as np
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from matchnet import MatchingNetwork


def sample_batch(data):
    """
    Generates sample batch
    :param : data - one of(train,test,val) our current dataset shape [total_classes,20,28,28,1]
    :return: [support_set_x,support_set_y,target_x,target_y] for Matching Networks
    """
    support_set_x = np.zeros((batch_size, n_way, k_shot, data.shape[2],
                              data.shape[3], data.shape[4]), np.float32)
    support_set_y = np.zeros((batch_size, n_way, k_shot), np.int32)

    target_x = np.zeros((batch_size, data.shape[2], data.shape[3], data.shape[4]), np.float32)
    target_y = np.zeros((batch_size, 1), np.int32)
    for i in range(batch_size):
        choose_classes = np.random.choice(data.shape[0], size=n_way, replace=False)  # choosing random classes
        choose_label = np.random.choice(n_way, size=1)  # label set
        choose_samples = np.random.choice(data.shape[1], size=k_shot + 1, replace=False)
        x_temp = data[choose_classes]  # choosing classes
        x_temp = x_temp[:, choose_samples]  # choosing sample batch from classes chosen outputs 20X2X28X28X1
        y_temp = np.arange(n_way)  # will return [0,1,2,3,...,19]
        support_set_x[i] = x_temp[:, :-1]
        support_set_y[i] = np.expand_dims(y_temp[:], axis=1)  # expand dimension
        target_x[i] = x_temp[choose_label, -1]
        target_y[i] = y_temp[choose_label]
    return support_set_x, support_set_y, target_x, target_y  # returns support of [batch_size, 20 classes per set, 1 sample, 28, 28,1]
def get_batch(dataset_name):
    """
    gen batch while training
    :param dataset_name: The name of dataset(one of "train","val","test")
    :return: a batch images
    """
    support_set_x, support_set_y, target_x, target_y = sample_batch(datatset[dataset_name])
    support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                           support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
    support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
    return support_set_x, support_set_y, target_x, target_y


if __name__ == '__main__':
    x = np.load(r'F:\jupyter_notebook\DAGAN\datasets\IITDdata_left.npy')  # Load Data
    print(x.shape) # (230, 6, 150, 150, 1)
    print(type(x))
    np.random.shuffle(x)  # shuffle dataset
    x_train, x_val  = x[:160], x[160:]  # divide dataset in to train, val,ctest
    # x_test =
    batch_size = 16  # setting batch_size
    n_classes = x.shape[0]  # total number of classes
    n_way = 5  # Number of classes per set
    k_shot = 3  # as we are choosing it to be one shot learning, so we have 1 sample

    def processes_batch(data, mu, sigma):
        return (data - mu) / sigma


    # Normalize Dataset
    x_train = processes_batch(x_train, np.mean(x_train), np.std(x_train))
    x_val = processes_batch(x_val, np.mean(x_val), np.std(x_val))
    # x_test = processes_batch(x_test, np.mean(x_test), np.std(x_test))

    # Defining dictionary of dataset
    # datatset = {"train": x_train, "val": x_val, "test": x_test}
    datatset = {"train": x_train, "val": x_val}
    num_channels = 1
    lr = 1e-3
    image_size = 150
    keep_prob = 0.0
    fce = True
    optim = "adam"
    wd = 0

    # matchNet = MatchingNetwork(keep_prob, batch_size, num_channels, lr, fce, n_way,
    #                            k_shot, image_size)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    argparser.add_argument('--k_shot', type=int, help='k shot for support set', default=3)  # default=1
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=16)

    argparser.add_argument('--keep_prob', type=int, help='keep_prob', default=0.0)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--image_size', type=int, help='image_size', default=150)  # 调节的图像尺寸
    argparser.add_argument('--num_channels', type=int, help='num_channels', default=1)
    argparser.add_argument('--fce', type=bool, help='fce', default=True)
    argparser.add_argument('--wd', type=int, help='wd', default=0)

    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    args = argparser.parse_args()
    matchNet = MatchingNetwork(args)


    total_iter = 0
    total_train_iter = 0
    optimizer = torch.optim.Adam(matchNet.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    # Training setup
    total_epochs = 20
    total_train_batches = 100
    total_val_batches = 10
    total_test_batches = 10

    def run_epoch(total_train_batches, name='train'):
        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0
        for i in range(int(total_train_batches)):
            x_support_set, y_support_set, x_target, y_target = get_batch(name)
            x_support_set = Variable(torch.from_numpy(x_support_set)).float()
            y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
            x_target = Variable(torch.from_numpy(x_target)).float()
            y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

            # convert to one hot encoding
            y_support_set = y_support_set.unsqueeze(2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = Variable(
                torch.zeros(batch_size, sequence_length,
                            n_way).scatter_(2, y_support_set.data, 1), requires_grad=False)

            # reshape channels and change order
            size = x_support_set.size()
            x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
            x_target = x_target.permute(0, 3, 1, 2)
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


    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    test_loss, test_accuracy = [], []

    for e in range(total_epochs):
        ############################### Training Step ##########################################
        total_c_loss, total_accuracy = run_epoch(total_train_batches, 'train')
        train_loss.append(total_c_loss)
        train_accuracy.append(total_accuracy)

        ################################# Validation Step #######################################
        total_val_c_loss, total_val_accuracy = run_epoch(total_val_batches, 'val')
        val_loss.append(total_val_c_loss)
        val_accuracy.append(total_val_accuracy)
        print("Epoch {}: train_loss:{:.2f} train_accuracy:{:.2f} valid_loss:{:.2f} valid_accuracy:{:.2f}".
              format(e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy))


    def plot_loss(train, val, name1="train_loss", name2="val_loss", title=""):
        plt.title(title)
        plt.plot(train, label=name1)
        plt.plot(val, label=name2)
        plt.legend()


    plot_loss(train_loss, val_loss, "train_loss", "val_loss", "Loss Graph")
    plot_loss(train_accuracy, val_accuracy, "train_accuracy", "val_accuracy", "Accuracy Graph")
