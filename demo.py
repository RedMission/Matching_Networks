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

def convLayer(in_channels, out_channels, dropout_prob=0.0):
    """
    :param dataset_name: The name of dataset(one of "train","val","test")
    :return: a batch images
    """
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(dropout_prob)
    )
    return cnn_seq
class Embeddings_extractor(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, dropout_prob=0.5, image_size=28):
        super(Embeddings_extractor, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.layer1 = convLayer(num_channels, layer_size, dropout_prob)
        self.layer2 = convLayer(layer_size, layer_size, dropout_prob)
        self.layer3 = convLayer(layer_size, layer_size, dropout_prob)
        self.layer4 = convLayer(layer_size, layer_size, dropout_prob)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

    def forward(self, image_input):
        """
        :param: Image
        :return: embeddings
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size()[0], -1)
        return x
class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Products pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
        :param support_set_y:[batch_size,sequence_length,classes_num]
        :return: Softmax pdf shape[batch_size,classes_num]
        """
        softmax = nn.Softmax(dim=1)
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds
class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and
    the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        forward pass
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()
class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size: 
        :param vector_dim: 
        """
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = (
        Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False),
        Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables,
        to detach them from their history."""
        if type(h) == torch.Tensor:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output


class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob, batch_size=32, num_channels=1, learning_rate=1e-3, fce=False, num_classes_per_set=20, \
                 num_samples_per_class=1, image_size=28):
        """
        Matching Network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        # Let's set all peices of Matching Networks Architecture
        self.g = Embeddings_extractor(layer_size=64, num_channels=num_channels, dropout_prob=keep_prob,
                                      image_size=image_size)
        self.f = fce  # if we are considering full-context embeddings
        self.c = DistanceNetwork()  # cosine distance among embeddings
        self.a = AttentionalClassify()  # softmax of cosine distance of embeddings
        if self.f:
            self.lstm = BidirectionalLSTM(layer_size=[32], batch_size=self.batch_size, vector_dim=self.g.outSize)

    def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:
        :return:
        """
        # produce embeddings for support set images
        encoded_images = []
        for i in np.arange(support_set_images.size(1)):
            gen_encode = self.g(support_set_images[:, i, :, :])
            encoded_images.append(gen_encode)

        # produce embeddings for target images
        gen_encode = self.g(target_image)
        encoded_images.append(gen_encode)
        output = torch.stack(encoded_images, dim=0)

        # if we are considering full-context embeddings
        if self.f:
            output = self.lstm(output)

        # get similarities between support set embeddings and target
        similarites = self.c(support_set=output[:-1], input_image=output[-1])

        # produce predictions for target probabilities
        preds = self.a(similarites, support_set_y=support_set_y_one_hot)

        # calculate the accuracy
        values, indices = preds.max(1)
        accuracy = torch.mean((indices.squeeze() == target_y).float())
        crossentropy_loss = F.cross_entropy(preds, target_y.long())

        return accuracy, crossentropy_loss





if __name__ == '__main__':
    x = np.load(r'F:\jupyter_notebook\DAGAN\datasets\IITDdata_left.npy')  # Load Data
    print(x.shape)
    print(type(x))
    x = np.reshape(x, newshape=(x.shape[0], x.shape[1], 150, 150, 1))  # expand dimension from (x.shape[0],x.shape[1],28,28)
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
    matchNet = MatchingNetwork(keep_prob, batch_size, num_channels, lr, fce, n_way,
                               k_shot, image_size)
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