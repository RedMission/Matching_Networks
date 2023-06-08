import numpy as np
import torch

def processes_batch(data, mu, sigma):
    return (data - mu) / sigma

def sample_batch(arg,dataset):
    """
    Generates sample batch
    :param : data - one of(train,test,val) our current dataset shape [total_classes,20,28,28,1]
    :return: [support_set_x,support_set_y,target_x,target_y] for Matching Networks
    """
    support_set_x = np.zeros((arg.batch_size, arg.n_way, arg.k_shot, dataset.shape[2],
                              dataset.shape[3], dataset.shape[4]), np.float32)
    support_set_y = np.zeros((arg.batch_size, arg.n_way, arg.k_shot), np.int32)

    target_x = np.zeros((arg.batch_size, dataset.shape[2], dataset.shape[3], dataset.shape[4]), np.float32)
    target_y = np.zeros((arg.batch_size, 1), np.int32)

    for i in range(arg.batch_size):
        choose_classes = np.random.choice(dataset.shape[0], size=arg.n_way, replace=False)  # choosing random classes
        choose_label = np.random.choice(arg.n_way, size=1)  # label set
        choose_samples = np.random.choice(dataset.shape[1], size=arg.k_shot + 1, replace=False)
        x_temp = dataset[choose_classes]  # choosing classes
        x_temp = x_temp[:, choose_samples]  # choosing sample batch from classes chosen outputs 20X2X28X28X1
        y_temp = np.arange(arg.n_way)  # will return [0,1,2,3,...,19]
        support_set_x[i] = x_temp[:, :-1]
        support_set_y[i] = np.expand_dims(y_temp[:], axis=1)  # expand dimension
        target_x[i] = x_temp[choose_label, -1]
        target_y[i] = y_temp[choose_label]
    return support_set_x, support_set_y, target_x, target_y  # returns support of [batch_size, 20 classes per set, 1 sample, 28, 28,1]

def get_batch(arg,dataset_name):
    """
    gen batch while training
    :param dataset_name: The name of dataset(one of "train","val","test")
    :return: a batch images
    """
    datatset = {"train": arg.train_data, "test": arg.test_data}
    # 加载原始数据
    dataset = np.load(datatset[dataset_name])
    np.random.shuffle(dataset)  # shuffle dataset

    # 数据归一化
    dataset = processes_batch(dataset, np.mean(dataset), np.std(dataset))

    support_set_x, support_set_y, target_x, target_y = sample_batch(arg,dataset)
    # support_set_x(16, 5, 3, 150, 150, 1)
    support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                           support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
    support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])

    return support_set_x, support_set_y, target_x, target_y







