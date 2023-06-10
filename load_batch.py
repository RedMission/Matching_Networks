import numpy as np
import cv2
import torch

def processes_batch(data, mu, sigma):
    return (data - mu) / sigma
def modify_support_image_size(dataset, new_size):
    modified_dataset = []
    for category in dataset:
        modified_category = []
        for sample in category:
            resized_image = cv2.resize(sample[:, :, 0], new_size)  # 只取图像的第一个通道进行调整
            resized_image = np.expand_dims(resized_image, axis=2)
            modified_category.append(resized_image)
        modified_dataset.append(modified_category)
    modified_dataset = np.array(modified_dataset)
    return modified_dataset
def modify_target_image_size(dataset, new_size):
    modified_dataset = []
    for category in dataset:
        resized_image = cv2.resize(category[:, :, 0], new_size)  # 只取图像的第一个通道进行调整
        resized_image = np.expand_dims(resized_image, axis=2)
        modified_dataset.append(resized_image)
    modified_dataset = np.array(modified_dataset)
    return modified_dataset

def sample_batch(arg,dataset):
    """
    Generates sample batch
    :param : data - one of(train,test,val) our current dataset shape [total_classes,20,28,28,1]
    :return: [support_set_x,support_set_y,target_x,target_y] for Matching Networks
    """
    support_set_x = np.zeros((arg.batch_size, arg.n_way, arg.k_shot, arg.image_size,
                              arg.image_size, dataset.shape[4]), np.float16)
    support_set_y = np.zeros((arg.batch_size, arg.n_way, arg.k_shot), np.int32)

    target_x = np.zeros((arg.batch_size, arg.image_size, arg.image_size, dataset.shape[4]), np.float16)
    target_y = np.zeros((arg.batch_size, 1), np.int32)

    for i in range(arg.batch_size):
        choose_classes = np.random.choice(dataset.shape[0], size=arg.n_way, replace=False)  # choosing random classes
        choose_label = np.random.choice(arg.n_way, size=1)  # label set
        choose_samples = np.random.choice(dataset.shape[1], size=arg.k_shot + 1, replace=False)
        x_temp = dataset[choose_classes]  # choosing classes
        x_temp = x_temp[:, choose_samples]  # choosing sample batch from classes chosen outputs 20X2X28X28X1
        y_temp = np.arange(arg.n_way)  # will return [0,1,2,3,...,19]
        # print(modify_image_size(x_temp[:, :-1], (arg.image_size,arg.image_size)).shape)
        support_set_x[i] = modify_support_image_size(x_temp[:, :-1], (arg.image_size,arg.image_size)) # 改尺寸
        support_set_y[i] = np.expand_dims(y_temp[:], axis=1)  # expand dimension
        target_x[i] = modify_target_image_size(x_temp[choose_label, -1], (arg.image_size,arg.image_size)) # 改尺寸
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

    # # 数据归一化
    # dataset = processes_batch(dataset, np.mean(dataset), np.std(dataset))

    support_set_x, support_set_y, target_x, target_y = sample_batch(arg,dataset)
    # support_set_x(batch_size, way, shot, h, w, c)
    support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                           support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
    support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
    del dataset
    return support_set_x, support_set_y, target_x, target_y







