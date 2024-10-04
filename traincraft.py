import os, argparse, time
import random
import numpy as np
from collections import OrderedDict
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from utils.imgproc import cvt2HeatmapImg

from models.architectures.model import Model_factory

###import file#######
from loader.craftloader import ListDataset
from loss.ohemloss import OHEMloss 
from utils.lr_scheduler import WarmupPolyLR
from utils.augmentations import Augmentation, Augmentation_test
    
cudnn.benchmark = True

def parse():

    # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./dataset/')
    parser.add_argument('--backbone', type=str, default='craft', help='[craft]')
    parser.add_argument('--batch_size', type=int, default=16, help='train batch size')
    parser.add_argument('--input_size', type=int, default=1024, help='input size')
    parser.add_argument('--workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--dataset', type=str, default='wapl', help='training dataset')
    parser.add_argument('--epochs', type=int, default=50, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=3.2768e-5, help='learning rate')
    parser.add_argument('--print_freq', default=100, type=int, help='interval of showing training conditions')
    parser.add_argument('--train_iter', default=0, type=int, help='number of total iterations for training')
    parser.add_argument('--curr_iter', default=0, type=int, help='current iteration')
    parser.add_argument('--save_path', type=str, default='./weight', help='Model save path')
    parser.add_argument('--resume', default="./weight/iter_loss.pth", type=str,  help='training restore')
    parser.add_argument('--weight_decay', default=5e-4, type=float,help='Weight decay for SGD')

    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--amp', action='store_true', help='half precision')

    args = parser.parse_args()

    return args

def main():

    args = parse()

    # fixed seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    mean=(0.485,0.456,0.406)
    var=(0.229,0.224,0.225)

    """ initial parameters for training """
    NUM_CLASSES = {'SynthText' : 1, 'handwritten': 4, 'generated': 2, 'wapl' : 1}
    num_classes = NUM_CLASSES[args.dataset]

    """ set model for training """
    model = Model_factory(args.backbone, num_classes)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    cudnn.benchmark = True
    device = torch.device('cuda:{}'.format(args.local_rank))

    # Scale learning rate based on global batch size
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = OHEMloss()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():

            if os.path.isfile(args.resume):
                if args.local_rank == 0: print("=> loading checkpoint '{}'".format(args.resume))
                model.load_state_dict(torch.load(args.resume))
                
                if args.local_rank == 0: print("=> loaded checkpoint ", args.resume)
            else:
                if args.local_rank == 0: print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    
    """ Get data loader """
    transform_train = Augmentation(args.input_size, mean, var)
    transform_valid = Augmentation_test(args.input_size, mean, var)

    train_dataset = ListDataset(root=args.root, dataset=args.dataset, input_size=args.input_size, mode='train', transform=transform_train)
    valid_dataset = ListDataset(root=args.root, dataset=args.dataset, input_size=args.input_size, mode='val', transform=transform_valid)

    if args.local_rank == 0: print("number of train = %d / valid = %d" % (len(train_dataset), len(valid_dataset)))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True, drop_last=False)
    
    """ lr scheduler """
    args.train_iter = len(train_loader) * args.epochs
    
    scheduler = WarmupPolyLR(
        optimizer,
        args.train_iter,
        warmup_iters=1000,
        power=0.90
    )
        
    if args.local_rank == 0: print(args)
    
    best_loss = 999999
    best_dist = 999999
    start = time.time()
    
    for epoch in range(0, args.epochs):
        # if distributed: train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, device, start, epoch, args)

        # evaluate on validation set
        val_loss = validate(valid_loader, model, criterion, device, epoch, args)

        # save checkpoint
        if args.local_rank == 0:

            if best_loss > val_loss:
                best_loss = val_loss
                model_file = os.path.join(args.save_path,  "best_loss.pth")
                torch.save(model.state_dict(), model_file)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
    
def train(train_loader, model, criterion, optimizer, scheduler, device, start, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

     # switch to train mode
    model.train()
    end = time.time()
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for images, region_labels, affinity_labels, confidence_masks in train_loader:
        args.curr_iter += 1
        
        images = images.to(device, non_blocking=True)
        region_labels = region_labels.to(device, non_blocking=True)
        affinity_labels = affinity_labels.to(device, non_blocking=True)
        confidence_masks = confidence_masks.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            out, _ = model(images)

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(out.cuda(), region_labels, affinity_labels)

        # Visualize training labels
        score_text = out1.cpu().data.numpy()
        score_label = region_labels.cpu().data.numpy()
        for i in range(len(score_text)):
            render_img = np.hstack((score_label[i], score_text[i]))
            ret_score_text = cvt2HeatmapImg(render_img)
            mask_file = "./output/" + str(i) + '_mask.jpg'
            cv2.imwrite(mask_file, ret_score_text)
 

        # compute gradient and backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        reduced_loss = loss.data
        losses.update(reduced_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.local_rank == 0 and args.curr_iter % args.print_freq == 0:
            train_log = "Epoch: [%d/%d][%d/%d] " % (epoch, args.epochs, args.curr_iter, args.train_iter)
            train_log += "({0:.1f}%, {1:.1f} min) | ".format(args.curr_iter/args.train_iter*100, (end-start) / 60)
            train_log += "Time %.1f ms | Left %.1f min | " % (batch_time.avg * 1000, (args.train_iter - args.curr_iter) * batch_time.avg / 60)
            train_log += "Loss %.6f " % (losses.avg)
            print(train_log)

        if args.curr_iter % 100 == 0:
            model_file = os.path.join(args.save_path,  "iter_loss.pth")
            torch.save(model.state_dict(), model_file)

                
    
def validate(valid_loader, model, criterion, device, epoch, args):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for images, region_labels, affinity_labels, confidence_masks in valid_loader:
        images = images.to(device, non_blocking=True)
        region_labels = region_labels.to(device, non_blocking=True)
        affinity_labels = affinity_labels.to(device, non_blocking=True)
        confidence_masks = confidence_masks.to(device, non_blocking=True)

        # compute output
        with torch.no_grad():
            out, _ = model(images)
            loss = criterion(out.cuda(), region_labels, affinity_labels)

        reduced_loss = loss.data

        losses.update(reduced_loss.item())

    if args.local_rank == 0:
        valid_log = "\n============== validation ==============\n"
        valid_log += "valid time : %.1f s | " % (time.time() - end)
        valid_log += "valid loss : %.6f | " % (losses.avg)
        print(valid_log)
        
    return losses.avg


def save_checkpoint(model, optimizer, epoch, name, save_path):
    state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
    model_file = os.path.join(save_path,  f"{name}.pth")
    torch.save(state, model_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    main()

    