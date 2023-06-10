import  torch
import  numpy as np
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from torch.utils.tensorboard import SummaryWriter
from npydataset import NpyDataset
from net import MatchingNetwork
from torch.optim.lr_scheduler import ReduceLROnPlateau



def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main():
    torch.manual_seed(122)  # 设置随机种子后，是每次运行文件的输出结果都一样
    torch.cuda.manual_seed_all(122)
    np.random.seed(122)
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MatchingNetwork(args).to(device) # 传入网络参数构建 maml网络
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print("maml:",maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode（一次选择support set和query set类别的过程） number
    train_data = NpyDataset(root = args.train_data,
                      mode='train', n_way=args.n_way,
                        k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=args.t_batchsz,  #
                        resize=args.imgsz,
                            imgc= args.imgc)
    test_data = NpyDataset(root = args.test_data, mode='test',
                             n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100,
                             resize=args.imgsz,
                            imgc= args.imgc)

    writer = SummaryWriter() # tensorboard
    for epoch in range(args.epoch//args.t_batchsz):  #
        print("epoch:",epoch)
        # db = DataLoader(train_data, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        db = DataLoader(train_data, args.task_num, shuffle=True, num_workers=2, pin_memory=True) # 生成可以将所有任务跑一遍的迭代器

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db): # 从迭代器取任务组合，每组完成一次外层循环，共step步外循环

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry[:, 0, ...].to(device), y_qry[:, 0].to(device)
            # [task_num,n*k,c,h,w],[task_num,n*k], 只用某类的一张来查询 [task_num,c,h,w]  [task_num,]
            accs,loss = net(x_spt, y_spt, x_qry, y_qry) # 传入的多个任务(共task_num个)

            # optimize process
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 可视化
            writer.add_scalar('train/loss',loss.item(), epoch*(args.t_batchsz//args.task_num)+step)
            writer.add_scalar('train/acc',accs.item(), epoch*(args.t_batchsz//args.task_num)+step)

            if step % 200 == 0:
                print('step:', step, '\t training acc:', accs.item())
                # evaluation
                db_test = DataLoader(test_data, args.task_num, shuffle=True, num_workers=2, pin_memory=True) # 测试 生成可以将所有任务跑一遍的迭代器
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), \
                                                 x_qry[:, 0, ...].to(device), y_qry[:, 0].to(device)

                    test_accs = net.finetunning(x_spt, y_spt, x_qry, y_qry) # [10,15,1,84,84]在测试中单独调用finetuning 返回精度
                    accs_all_test.append(test_accs.item())

                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16) # 求测试数据的均值
                writer.add_scalar('test/acc', accs.item(), epoch * (args.t_batchsz // args.task_num) + step)
                print('Test acc:', accs)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_data', type=str, help='', default='F:\jupyter_notebook\DAGAN\datasets\IITDdata_left.npy')
    argparser.add_argument('--test_data', type=str, help='', default='F:\jupyter_notebook\DAGAN\datasets\PolyUROI.npy')

    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)

    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3) # default=1
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1) # 原15
    argparser.add_argument('--t_batchsz', type=int, help='train-batchsz', default=5000)

    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84) # 调节的图像尺寸
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-3)

    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    ############
    argparser.add_argument('--keep_prob', type=int, help='keep_prob', default=0.0)
    argparser.add_argument('--device', type=str, help='device', default="cuda")
    argparser.add_argument('--fce', type=bool, help='fce', default=True)
    argparser.add_argument('--wd', type=int, help='wd', default=0)

    args = argparser.parse_args()

    main()
