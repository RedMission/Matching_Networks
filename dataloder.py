import numpy as np
import matplotlib.pyplot as plt



def processes_batch(data, mu, sigma):
    return (data - mu) / sigma


def sample_batch(data,batch_size,n_way,k_shot): # 取消k_query
    """
    Generates sample batch
    :param : data - one of(train,test,val) our current dataset shape [total_classes,20,28,28,1]
    :return: [support_set_x,support_set_y,target_x,target_y] for Matching Networks, support_set_x=[batchsize,way,shot,h,w,c] target_x=[batchsize,k_query,h,w,c]
    """
    support_set_x = np.zeros((batch_size, n_way, k_shot, data.shape[2],
                              data.shape[3], data.shape[4]), np.float32)
    support_set_y = np.zeros((batch_size, n_way, k_shot), np.int32)

    target_x = np.zeros((batch_size, data.shape[2], data.shape[3], data.shape[4]), np.float32)
    target_y = np.zeros((batch_size), np.int32)

    for i in range(batch_size):
        choose_classes = np.random.choice(data.shape[0], size=n_way, replace=False)  # choosing random classes
        choose_label = np.random.choice(n_way, size=1)  # label set
        choose_samples = np.random.choice(data.shape[1], size=k_shot, replace=False)
        x_temp = data[choose_classes]  # choosing classes
        x_temp = x_temp[:, choose_samples]  # choosing sample batch from classes chosen
        y_temp = np.arange(n_way)  # 重新设定标签 will return [0,1,2,3,...,19]
        aaa = x_temp[:, :k_shot]
        support_set_x[i] = x_temp[:, :-1]  # 只保留一部分作为支持集
        support_set_y[i] = np.expand_dims(y_temp[:], axis=1)  # expand dimension
        target_x[i] = x_temp[choose_label, -1]  # 剩下部分作为查询集
        target_y[i] = y_temp[choose_label]
    return support_set_x, support_set_y, target_x, target_y

def get_batch(datatset, dataset_name,batch_size,n_way,k_shot): # 调整way*shot 和通道位置
    """
    gen batch while training
    :param dataset_name: The name of dataset(one of "train","val","test")
    :return: a batch images
    """
    support_set_x, support_set_y, target_x, target_y = sample_batch(datatset[dataset_name],batch_size,n_way,k_shot)
    support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                           support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
    support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
    return support_set_x, support_set_y, target_x, target_y


if __name__ == '__main__':
    x = np.load('F:\jupyter_notebook\DAGAN\datasets\IITDdata_left.npy')  # Load Data
    # print(x.shape) #(230, 6, 150, 150, 1)
    np.random.shuffle(x)  # shuffle dataset
    x_train, x_val = x[:160], x[160:]  # divide dataset in to train, val
    batch_size = 16  # 多个任务组成的任务集合
    n_classes = x.shape[0]  # total number of classes
    k_shot = 3 # shot
    k_query = 2
    n_way = 5

    # Normalize Dataset
    x_train = processes_batch(x_train, np.mean(x_train), np.std(x_train))
    x_val = processes_batch(x_val, np.mean(x_val), np.std(x_val))

    # Defining dictionary of dataset
    datatset = {"train": x_train, "val": x_val}

    #######
    support_set_x, support_set_y, target_x, target_y = get_batch(datatset,"train",batch_size,n_way,k_shot,k_query)
    print(support_set_x)


