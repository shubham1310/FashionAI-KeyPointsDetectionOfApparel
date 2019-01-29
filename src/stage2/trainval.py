import os
import time
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from tensorboard_logger import configure, log_value

import sys
sys.path.append('/home/nfs/shubham9/keypointdetect')

from src import pytorch_utils
from src.kpda_parser import KPDA
from src.config import Config
from src.stage2.data_generator import DataGenerator
from src.stage2.cascade_pyramid_network import CascadePyramidNet
from src.stage2v9.cascade_pyramid_network_v9 import CascadePyramidNetV9
from src.stage2.viserrloss import VisErrorLoss
# from src.lr_scheduler import LRScheduler


# def print_log(epoch, lr, train_metrics, train_time, val_metrics=None, val_time=None, save_dir=None, log_mode=None):
#     if epoch > 1:
#         log_mode = 'a'
#     train_metrics = np.mean(train_metrics, axis=0)
#     str0 = 'Epoch %03d (lr %.7f)' % (epoch, lr)
#     str1 = 'Train:      time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f' \
#            % (train_time, train_metrics[0], train_metrics[1], train_metrics[2])
#     print(str0)
#     print(str1)
#     f = open(save_dir + 'kpt_' + config.clothes + '_train_log.txt', log_mode)
#     f.write(str0 + '\n')
#     f.write(str1 + '\n')
#     if val_time is not None:
#         val_metrics = np.mean(val_metrics, axis=0)
#         str2 = 'Validation: time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f' \
#                % (val_time, val_metrics[0], val_metrics[1], val_metrics[2])
#         print(str2 + '\n')
#         f.write(str2 + '\n\n')
#     f.close()

def train(data_loader, net, loss, optimizer, lr, epoch):
    start_time = time.time()

    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, heatmaps, vismaps) in enumerate(data_loader):
        data = data.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        vismaps = vismaps.cuda(async=True)
        heat_pred1, heat_pred2 = net(data)
        loss_output = loss(heatmaps, heat_pred1, heat_pred2, vismaps)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()
        metrics.append([loss_output[0].item(), loss_output[1].item(), loss_output[2].item()])
        if (i%50==0):
            print("%d/%d loss:%.3f"%(i,len(data_loader),loss_output[0].data))
            log_value('iteration loss', loss_output[0].data, i+epoch*len(data_loader))
    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time


def validate(data_loader, net, loss):
    start_time = time.time()
    net.eval()
    metrics = []
    for i, (data, heatmaps, vismaps) in enumerate(data_loader):
        data = data.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        vismaps = vismaps.cuda(async=True)
        heat_pred1, heat_pred2 = net(data)
        loss_output = loss(heatmaps, heat_pred1, heat_pred2, vismaps)
        metrics.append([loss_output[0].item(), loss_output[1].item(), loss_output[2].item()])
    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--clothes', help='specify the clothing type', default='outwear')
    parser.add_argument('-r', '--resume', help='specify the checkpoint', default=None)
    parser.add_argument('--data_path', type=str, default='/home/nfs/shubham9/keypointdetect/', help='project path')
    parser.add_argument('--batchsize', type=int, default=6, help='batchsize per gpu')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--img_max_size', type=int, default=512, help='maximum image size')
    parser.add_argument('--mu', type=float, default=0.65, help='mu')
    parser.add_argument('--sigma', type=float, default=0.25, help='sigma ')
    parser.add_argument('--gpus', type=str, default='0', help='gpus')
    parser.add_argument('--out', type=str, default='stage2/', help='output directory')
    parser.add_argument('--train', type=int, default=0, help='train 0/val 1')


    args = parser.parse_args(sys.argv[1:])
    print(args)
    print('Training ' + args.clothes)

    data_path = args.data_path + 'FashionAIdataset/'

    config = Config(args)
    workers = config.workers
    n_gpu = pytorch_utils.setgpu(args.gpus)
    batch_size = args.batchsize * n_gpu

    epochs = args.nEpochs
    # 256 pixels: SGD L1 loss starts from 1e-2, L2 loss starts from 1e-3
    # 512 pixels: SGD L1 loss starts from 1e-3, L2 loss starts from 1e-4
    base_lr = args.lr
    save_dir = 'log/' + args.out
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    configure('log/' + args.out, flush_secs=5)

    net = CascadePyramidNet(config)
    loss = VisErrorLoss()
    train_data = KPDA(config, data_path, 'train')
    val_data = KPDA(config, data_path, 'val')
    print('Train sample number: %d' % train_data.size())
    print('Val sample number: %d' % val_data.size())

    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    # log_mode = 'w'
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        # start_epoch = checkpoint['epoch'] + 1
        # lr = checkpoint['lr']
        # best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        # log_mode = 'a'

    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)

    train_dataset = DataGenerator(config, train_data, phase='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
    val_dataset = DataGenerator(config, val_data, phase='val')
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            collate_fn=val_dataset.collate_fn,
                            pin_memory=True)
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)

    if args.train==0:

        # lrs = LRScheduler(lr, patience=3, factor=0.1, min_lr=0.01*lr, best_loss=best_val_loss)
        for epoch in range(start_epoch, epochs + 1):
            train_metrics, train_time = train(train_loader, net, loss, optimizer, lr, epoch)
            with torch.no_grad():
                val_metrics, val_time = validate(val_loader, net, loss)

            # print_log(epoch, lr, train_metrics, train_time, val_metrics, val_time, save_dir=save_dir, log_mode=log_mode)
            train_metrics = np.array(train_metrics)
            n= len(train_metrics)
            log_value('train loss',  sum(train_metrics[:,0])/n, epoch)
            log_value('train loss 1',  sum(train_metrics[:,0])/n, epoch)
            log_value('train loss 2',  sum(train_metrics[:,0])/n, epoch)
            log_value('train time',  train_time, epoch)
            val_metrics = np.array(val_metrics)
            n= len(val_metrics)
            log_value('validation loss',  sum(val_metrics[:,0])/n, epoch)
            log_value('validation loss 1',  sum(val_metrics[:,1])/n, epoch)
            log_value('validation loss 2',  sum(val_metrics[:,2])/n, epoch)
            log_value('validation time',  val_time, epoch)

            val_loss = np.mean(val_metrics[:, 0])
            # lr = lrs.update_by_rule(val_loss)
            if val_loss < best_val_loss or epoch%4 == 0 or lr is None:
                state_dict = net.module.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    log_value('best validation loss',  best_val_loss, epoch)
                    torch.save({
                        'state_dict': state_dict,
                        'best_val_loss': best_val_loss},
                        os.path.join(save_dir, 'kpt_'+config.clothes+'_best_%03d.ckpt' % epoch))
                torch.save({
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict': state_dict,
                    'lr': lr,
                    'best_val_loss': best_val_loss},
                    os.path.join(save_dir, 'kpt_'+config.clothes+'_%03d.ckpt' % epoch))
            if lr is None:
                print('Training is early-stopped')
                # print('best_val_loss %f'%(best_val_loss))
                break


    else:
        with torch.no_grad():
            val_metrics, val_time = validate(val_loader, net, loss)
        val_metrics = np.array(val_metrics)
        n= len(val_metrics)
        print('Validation: time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f \n' \
               % (val_time, sum(val_metrics[:,0])/n, sum(val_metrics[:,1])/n, sum(val_metrics[:,2])/n))



