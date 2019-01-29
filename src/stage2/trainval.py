import os
import time
import argparse
import numpy as np
import torch
import cv2
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
from src.stage2.data_generator import DataGenerator, transform
from src.stage2.cascade_pyramid_network import CascadePyramidNet
from src.stage2v9.cascade_pyramid_network_v9 import CascadePyramidNetV9
from src.stage2.viserrloss import VisErrorLoss


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
    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')


    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])

    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    
    if args.train<=1:
        data_path = args.data_path + 'FashionAIdataset/'
        val_data = KPDA(config, data_path, 'val')
        print('Val sample number: %d' % val_data.size())
        val_dataset = DataGenerator(config, val_data, phase='val')
        val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            collate_fn=val_dataset.collate_fn,
                            pin_memory=True)

        if args.train==0:
            train_data = KPDA(config, data_path, 'train')
            print('Train sample number: %d' % train_data.size())
            train_dataset = DataGenerator(config, train_data, phase='train')
            train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
            optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
            # optimizer = optim.Adam(net.parameters(),lr = lr )
            for epoch in range(start_epoch, epochs + 1):
                train_metrics, train_time = train(train_loader, net, loss, optimizer, lr, epoch)
                with torch.no_grad():
                    val_metrics, val_time = validate(val_loader, net, loss)

                train_metrics = np.array(train_metrics)
                n= len(train_metrics)
                log_value('train loss',  np.mean(train_metrics[:,0]), epoch)
                log_value('train loss 1',  np.mean(train_metrics[:,0]), epoch)
                log_value('train loss 2',  np.mean(train_metrics[:,0]), epoch)
                log_value('train time',  train_time, epoch)
                val_metrics = np.array(val_metrics)
                n= len(val_metrics)
                log_value('validation loss',  np.mean(val_metrics[:,0]), epoch)
                log_value('validation loss 1',  np.mean(val_metrics[:,1]), epoch)
                log_value('validation loss 2',  np.mean(val_metrics[:,2]), epoch)
                log_value('validation time',  val_time, epoch)

                val_loss = np.mean(val_metrics[:, 0])
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
                    break


        else :
            with torch.no_grad():
                val_metrics, val_time = validate(val_loader, net, loss)
            val_metrics = np.array(val_metrics)
            n= len(val_metrics)
            print('Validation: time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f \n' \
                   % (val_time, sum(val_metrics[:,0])/n, sum(val_metrics[:,1])/n, sum(val_metrics[:,2])/n))
    else:
        img = cv2.imread(args.data_path)
        img = transform(img, args.img_max_size, args.mu, args.sigma)


