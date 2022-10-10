import argparse
import os

import ray
import torch
import torch.nn as nn
from ray import tune

from datasets import Digits
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Teacher Network training code')
parser.add_argument('--dataset_dir', type=str, default='/home/chen_h/datasets/Digits', help="directory of the dataset.")
parser.add_argument('--domain', type=str, help="domain for training and testing")
parser.add_argument('--mode', type=str, default='train', help="mode of script.")
parser.add_argument('--img_size', type=int, default=32, help="input image size.")
parser.add_argument('--split', type=bool, default=True, help="whether to split dataset.")
parser.add_argument('--batch_size', type=int, default=128, help="batch size of training")
opt = parser.parse_args()

def train(opt):
    best = 0
    best_ep = 0
    trainset = Digits(opt.dataset_dir, mode = 'train', img_size=opt.img_size, domain=opt.domain)
    valset = Digits(opt.dataset_dir, mode = 'val', img_size=opt.img_size, domain=opt.domain)

    trainset_loader = torch.utils.data.DataLoader(dataset=trainset, shuffle=True, batch_size=opt.batch_size, collate_fn=trainset.collate_fn)
    valset_loader = torch.utils.data.DataLoader(dataset=valset, shuffle=True, batch_size=opt.batch_size, collate_fn=valset.collate_fn)
    #########Here start from ImageNet pretrained network.##################
    net = get_net(opt.backbone, len(trainset.classes), pretrained=True)
    net = net.cuda().train()
    #net.apply(set_bn_eval)

    snapshot_dir = os.path.realpath('/home/chen_h/DFDG/saved_models_teacher/Digits/' + opt.domain + '_' + opt.backbone)
    #net.load_state_dict(torch.load(snapshot_dir + '/model_best.pth'))

    if not os.path.isdir(snapshot_dir):
        os.system(f'mkdir -p {snapshot_dir}')

    max_ep = 50
#     optimizer = torch.optim.Adam(net.parameters(), opt.lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    l_criterion = nn.CrossEntropyLoss()

    lr_scheduler = lr_cosine_policy(opt.lr, min(4, max_ep*0.1), max_ep)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.9)

    for ep in range(max_ep):
        cur_lr = lr_scheduler(optimizer, ep, ep)
        for (imgs, labels) in trainset_loader:
            optimizer.zero_grad()
            logits = net(imgs)
            loss = l_criterion(logits, labels.squeeze(-1))
            loss.backward()
            optimizer.step()
            tune.report(loss = float(loss))
        tune.report()
        #scheduler.step()

        test_acc = evaluate(net, valset_loader)
        if test_acc>best:
            torch.save(net.state_dict(), snapshot_dir + f'/model_best_{opt.lr}_{opt.batch_size}.pth')
            best = test_acc
            best_ep = ep

        if test_acc<best and ep-best_ep>5 and ep>int(max_ep*0.8):
            tune.report( best_acc = best )
            print(opt.backbone, opt.domain, best)
            tune.report( done = True)

        tune.report( test_acc = float(test_acc),
                     lr = float(cur_lr)
                   )

    print(opt.backbone, opt.domain, best)
    tune.report( best_acc = best )

def train_one_setting(config):

    opt.lr = config['lr']
    opt.backbone = config['backbone']
    opt.domain = config['domain']
    opt.batch_size = config['batch_size']

    train(opt)

def main():
    setup_seed(123)

    snapdir = '/home/chen_h/dfdg'
    ray.init()

    search_lr = [1e-2, 5e-3]
    search_batch_size = [1024, 2048]
    search_backbone = ['resnet18']
    search_domains = ['SVHN']
    tune.run(
        train_one_setting,
        config={
            "lr": tune.grid_search(search_lr),
            "backbone": tune.grid_search(search_backbone),
            'domain': tune.grid_search(search_domains),
            'batch_size': tune.grid_search(search_batch_size),
        },
        resources_per_trial={"gpu": 0.5, "cpu": 2},
        local_dir=snapdir + "/logs_teacher_train",
    )

if __name__=='__main__':
    main()
