import  argparse
from torch.autograd import Variable
import numpy as np
import torch

from matchnet import MatchingNetwork

import matplotlib.pyplot as plt

from load_batch import get_batch


def plot_loss(train, val, name1="train_loss", name2="val_loss", title=""):
    plt.title(title)
    plt.plot(train, label=name1)
    plt.plot(val, label=name2)
    plt.legend()

def main():
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化网络对象
    matchNet = MatchingNetwork(args).to(device)
    tmp = filter(lambda x: x.requires_grad, matchNet.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)
    # 参数优化器
    optimizer = torch.optim.Adam(matchNet.parameters(), lr=args.lr, weight_decay=args.wd)

    # 迭代
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    def run_epoch(args, name):
        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0

        for i in range(int(args.total_train_batches)):
            x_support_set, y_support_set, x_target, y_target = get_batch(args, name)
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
                            args.n_way).scatter_(2, y_support_set.data, 1), requires_grad=False)

            # reshape channels and change order
            size = x_support_set.size()
            x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
            x_target = x_target.permute(0, 3, 1, 2)

            x_support_set, y_support_set_one_hot, x_target, y_target = x_support_set.to(device), y_support_set_one_hot.to(device), \
                                                                    x_target.to(device), y_target.to(device)

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

        total_c_loss = total_c_loss / args.total_train_batches
        total_accuracy = total_accuracy / args.total_train_batches
        return total_c_loss, total_accuracy
    for e in range(args.total_epochs):
        ############################### Training Step ##########################################
        total_c_loss, total_accuracy = run_epoch(args,'train')
        train_loss.append(total_c_loss)
        train_accuracy.append(total_accuracy)
    ################################# Validation Step #######################################
        total_val_c_loss, total_val_accuracy = run_epoch(args, 'test')
        val_loss.append(total_val_c_loss)
        val_accuracy.append(total_val_accuracy)
        print("Epoch {}: train_loss:{:.2f} train_accuracy:{:.2f} valid_loss:{:.2f} valid_accuracy:{:.2f}".
              format(e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy))

    plot_loss(train_loss, val_loss, "train_loss", "val_loss", "Loss Graph")
    plot_loss(train_accuracy, val_accuracy, "train_accuracy", "val_accuracy", "Accuracy Graph")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_data', type=str, help='',
                           default='F:\jupyter_notebook\DAGAN\datasets\IITDdata_left.npy')
    argparser.add_argument('--test_data', type=str, help='',
                           default='F:\jupyter_notebook\DAGAN\datasets\IITDdata_right.npy')

    argparser.add_argument('--n_way', type=int, help='n way', default=5)

    argparser.add_argument('--k_shot', type=int, help='k shot for support set', default=3)  # default=1
    argparser.add_argument('--t_batchsz', type=int, help='train-batchsz', default=5000)
    argparser.add_argument('--batch_size', type=int, help='一个任务集合中任务的个数', default=4)

    argparser.add_argument('--keep_prob', type=int, help='keep_prob', default=0.0)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--image_size', type=int, help='image_size', default=84)  # 图像尺寸——用于设定网络结构，需要设定调节data尺寸
    argparser.add_argument('--num_channels', type=int, help='num_channels', default=1)
    argparser.add_argument('--fce', type=bool, help='fce', default=True)
    argparser.add_argument('--wd', type=int, help='wd', default=0)

    argparser.add_argument('--total_epochs', type=int, help='total_epochs number', default=20)
    argparser.add_argument('--total_train_batches', type=int, help='total_train_batches number', default=100)
    argparser.add_argument('--total_val_batches', type=int, help='total_val_batches number', default=10)
    argparser.add_argument('--total_test_batches', type=int, help='total_test_batches number', default=10)

    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)

    args = argparser.parse_args()
    main()