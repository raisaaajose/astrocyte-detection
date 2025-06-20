import os
import sys
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook


class Astrocytes(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        return image

def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j)) in enumerate(train_loader):
        
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)
        #positive pair, with encoding
        
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        
        loss = criterion(z_i, z_j)
        try:
            loss.backward()
        except RuntimeError as e:
            print("Error during backward:", e)
            raise e
        optimizer.step()
        if dist.is_available() and dist.is_initialized():
            with torch.no_grad():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))
        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1
        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    print("main function started")
    args.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu)

    args.local_rank = gpu
    rank = args.nr * args.gpus + gpu
    print("test1")
    if args.nodes >= 1:
        print("rank "+str(rank)+" world_size "+str(args.world_size))
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        
        torch.cuda.set_device(gpu)
    print("test")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "astrocytes":
        print("Loading Astrocytes dataset")
        
        train_dataset = Astrocytes(
            root_dir=args.dataset_dir,
            transform=TransformsSimCLR(size=args.image_size)
        )
        
    else:
        raise NotImplementedError
    
    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
        
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=4,
        sampler=train_sampler,
    )
    print("DataLoader initialized")
    
    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    print("resnet initialized with {} features".format(n_features))
    
    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)
    
    # DDP / DP
    if args.dataparallel:
        
        model = convert_model(model)
        
        model = DataParallel(model)
        
    else:
        
        # if args.nodes > 1:
            
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DDP(model, device_ids=[gpu])
            
    
    
    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()
    
    args.global_step = 0
    args.current_epoch = 0
    print("start:"+str(args.start_epoch) +"stop: "+str(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        print("starting epoch {}".format(epoch))
        lr = optimizer.param_groups[0]["lr"]
        
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)
        print("training done")
        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            print("model saved at epoch {}".format(epoch))
            
            save_model(args, model, optimizer)

        if args.nr == 0:
            print("model saved at epoch {}".format(epoch))
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1
    
    print("training finished")
    
    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"
    
 
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print("GPU is available")

    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes >= 1:
        
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        
        main(0, args)
    

