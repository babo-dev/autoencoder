import os
import time
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import utils
from models import builder
import dataloader


def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='simple', type=str,
                        help='backbone architechture')
    parser.add_argument('--train_list', type=str)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=2, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--pth-save-fold', default='results/tmp', type=str,
                        help='The folder to save pths')
    parser.add_argument('--pth-save-epoch', default=2, type=int,
                        help='The epoch to save pth')
    parser.add_argument('--parallel', type=int, default=1,
                        help='1 for parallel, 0 for non-parallel')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    # ngpus_per_node = torch.cuda.device_count()
    # print('=> ngpus : {}'.format(ngpus_per_node))
    args.world_size = 1
    main_worker(args)


def main_worker(args):
    device = 'cpu'
    model = builder.BuildAutoEncoder(args.arch).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024 * 1024))))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                 weight_decay=args.weight_decay)
    train_loader = dataloader.train_loader(args)

    criterion = nn.MSELoss()

    global iters
    iters = 0

    outputs = []
    model.train()
    for epoch in range(args.start_epoch, args.epochs):

        global current_lr
        current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args.lr, args.epochs)

        # train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        epoch_outputs = do_train(train_loader, model, criterion, optimizer, epoch, args)
        outputs.extend(epoch_outputs)

        # save pth
        if epoch % args.pth_save_epoch == 0:
            state_dict = model.state_dict()

            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(args.pth_save_fold, 'l{}.pth'.format(str(epoch).zfill(3)))
            )

            print(' : save pth for epoch {}'.format(epoch + 1))

    for k in range(0, args.epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9:
                break
            plt.subplot(2, 9, i + 1)
            # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
            # item: 1, 28, 28
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9 + i + 1)  # row_length + i + 1
            # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
            # item: 1, 28, 28
            plt.imshow(item[0])


def do_train(train_loader, model, criterion, optimizer, epoch, args) -> list:
    batch_time = utils.AverageMeter('Time', ':6.2f')
    data_time = utils.AverageMeter('Data', ':2.2f')
    losses = utils.AverageMeter('Loss', ':.4f')
    learning_rate = utils.AverageMeter('LR', ':.4f')

    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate],
        prefix="Epoch: [{}]".format(epoch + 1))
    end = time.time()

    # update lr
    learning_rate.update(current_lr)

    outputs = []
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1

        # input = input.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        output = model(input)

        loss = criterion(output, target)

        # compute gradient and do solver step
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weights
        optimizer.step()

        # syn for logging
        # torch.cuda.synchronize()

        # record loss
        losses.update(loss.item(), input.size(0))

        # # measure elapsed time
        # if args.rank == 0:
        batch_time.update(time.time() - end)
        end = time.time()

        outputs.append((epoch, input, output))
        if i % args.print_freq == 0:
            progress.display(i)

    return outputs


if __name__ == '__main__':
    args = get_args()
    args.train_list = "/list/101_object_list.txt"
    args.pth_save_fold = "/results"
    args.local = True

    main(args)
