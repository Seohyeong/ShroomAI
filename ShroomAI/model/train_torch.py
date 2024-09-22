# -*- coding: utf-8 -*-
# ref: https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py

import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
# plt.ion()   # interactive mode

import datetime
import json
from tqdm import tqdm

# # mlflow
# import mlflow
# from mlflow.models import infer_signature

def custom_print(message, file_path='training_log.txt'):
    with open(file_path, 'a') as f:
        f.write(message + '\n')
    print(message)
    
    
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_model(model, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs, device, 
                early_stopping=None, log_file=None):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                if phase == 'train':
                    dataloader = tqdm(dataloaders[phase], 
                                    total=len(dataloaders[phase]), 
                                    desc=f' >> [{phase}] Epoch {epoch+1}/{num_epochs}')
                else:
                    dataloader = dataloaders[phase]
                    
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'val':
                    scheduler.step(epoch_acc) # TODO: scheduler update in val loop
                    if early_stopping:
                        early_stopping(epoch_acc)
                        if early_stopping.early_stop:
                            custom_print(" >> Early stopping!", log_file)
                            model.load_state_dict(best_model_wts)
                            return model

                log_msg = (
                    f' >> [Epoch {epoch + 1}/{num_epochs}] '
                    f'    {"Train" if phase == "train" else "Val  "} Loss: {epoch_loss:.4f} '
                    f'    {"Train" if phase == "train" else "Val  "} Acc: {epoch_acc:.4f} '
                    f'    {"Current lr: " + str(optimizer.param_groups[0]["lr"]) if phase == "train" else ""}'
                )
                custom_print(log_msg, log_file)
                
                # # log to mlflow: loss, acc, lr
                # # num_step in one epoch = (total_dataset_size / batch_size) * (epoch + 1)
                # #                       = len(dataloaders[phase]) * (epoch + 1)
                # mlflow.log_metric(key=f'{"train" if phase == "train" else "val"}_loss', 
                #                   value=f"{epoch_loss:4f}", 
                #                   step=len(dataloaders[phase])*(epoch+1))
                # mlflow.log_metric(key=f'{"train" if phase == "train" else "val"}_acc', 
                #                   value=f"{epoch_acc:4f}", 
                #                   step=len(dataloaders[phase])*(epoch+1))
                # if phase == 'train':
                #     mlflow.log_metric(key='lr', value=optimizer.param_groups[0]['lr'], step=len(dataloaders[phase])*(epoch+1))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_model_params_path)
        
        time_elapsed = time.time() - since
        custom_print(f' >> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Best val Acc: {best_acc:4f}',
                     log_file)

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


def eval_model(model, dataloaders, dataset_sizes, device, log_file):
    model.eval()

    running_loss = 0.0
    running_corrects_topn = [0] * 5

    with torch.set_grad_enabled(False):
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            _, topk_indices = torch.topk(outputs, 5, dim=1)

            for k in range(1, 6):
                correct_topk = topk_indices[:, :k].eq(labels.view(-1, 1).expand_as(topk_indices[:, :k]))
                running_corrects_topn[k-1] += correct_topk.sum().item()

            running_loss += loss.item() * inputs.size(0)

    loss = running_loss / dataset_sizes['val']
    topn_accuracies = [corrects / dataset_sizes['val'] for corrects in running_corrects_topn]
    
    formatted_strs = [' >> top {}: {:.4f}    '.format(n + 1, acc) for n, acc in enumerate(topn_accuracies)]
    custom_print(''.join(formatted_strs), log_file)
    
    return (loss, topn_accuracies)


def save_model(model, optimizer, file_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)


def main():
    parser = argparse.ArgumentParser(description='Training MobileNetV2 with GBIF Mushroom Dataset')

    # path
    parser.add_argument('--dataset_dir_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/dataset/images_100_3314')
    parser.add_argument('--ckpt_dir_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/ckpt')
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='pass saved model path to run evaluation')
    parser.add_argument('--pretrain_model_path', type=str,
                        default=None,
                        help='continue finetuning with the pretrained checkpoint')

    # lr decay
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--cooldown', type=int, default=0)
    parser.add_argument('--min_delta', type=float, default=0.01)
    parser.add_argument('--factor', type=float, default=0.25)

    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='evaluation batch size') # 1024

    # pretrain
    parser.add_argument('--pretrain_epoch', type=int, default=20, help='epoch for finetuning classification head') # 20
    parser.add_argument('--pretrain_batch_size', type=int, default=1024, help='training batch size')
    parser.add_argument('--pretrain_learning_rate', type=float, default=0.0005) # 0.0005

    # finetune
    parser.add_argument('--finetune_epoch', type=int, default=40, help='epoch for full finetuning') # 40
    parser.add_argument('--finetune_batch_size', type=int, default=512, help='training batch size')
    parser.add_argument('--finetune_learning_rate', type=float, default=0.00001) # 0.0001

    # options
    parser.add_argument('--pretrain', action='store_true', help='classification head training')
    parser.add_argument('--finetune', action='store_true', help='full finetuning')
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    if not args.pretrain and args.finetune:
        assert args.pretrain_model_path, "Pretrain first to finetune, otherwise pass pretrain_model_path."

    # Dataloaders
    print('> Building Dataloader...')
    data_transforms = {
        # transforms.Normalize(mean=[0.5], std=[0.5] # [-1, 1] normalization
        'train': transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.dataset_dir_path, x),
                                              data_transforms[x],
                                              allow_empty=True)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    val_dataloader = torch.utils.data.DataLoader(image_datasets['val'], 
                                                 batch_size=args.eval_batch_size, 
                                                 shuffle=False, 
                                                 num_workers=4)
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(' >> # class: {}, # train samples: {}, # val samples: {}'.format(
        num_classes, dataset_sizes['train'], dataset_sizes['val']))

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # setup mlflow
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    # mlflow.set_experiment("ShroomAI")
    
    # # Start an MLflow run
    # with mlflow.start_run():
    #     params = args.kwargs
    #     params.update({'device': device})
    #     mlflow.log_params(args.kwargs)

    #     # Set a tag that we can use to remind ourselves what this run was for
    #     mlflow.set_tag("Training Info", "MobileNetV2 with mushroom dataset")
        
            
    if args.pretrain:
        
        save_model_dir_path = os.path.join(args.ckpt_dir_path, 'model_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        os.mkdir(save_model_dir_path)
        
        log_file = os.path.join(save_model_dir_path, "log.txt")
        
        custom_print('> Loading Pre-trained MobileNetV2...', log_file)
        mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1') # weights='IMAGENET1K_V1'
        in_features = mobilenet.classifier[1].in_features
        mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        mobilenet = mobilenet.to(device)

        # Pretraing
        optimizer_pt = optim.RMSprop(mobilenet.classifier.parameters(), lr=args.pretrain_learning_rate)
        scheduler_pt = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_pt,
                                                mode='max', 
                                                threshold=args.min_delta,
                                                patience=args.patience,
                                                factor=args.factor,
                                                cooldown=args.cooldown,
                                                min_lr=0.00000001)
        early_stopping = EarlyStopping(patience=10, min_delta=0.005)
        
        pt_train_dataloader = torch.utils.data.DataLoader(image_datasets['train'], 
                                                        batch_size=args.pretrain_batch_size, 
                                                        shuffle=True, 
                                                        num_workers=4)
        pt_dataloaders = {'train': pt_train_dataloader, 'val': val_dataloader}

        custom_print('> Training Classification Head...', log_file)
        mobilenet_pt = train_model(model=mobilenet,
                                dataloaders=pt_dataloaders,
                                dataset_sizes=dataset_sizes,
                                optimizer=optimizer_pt,
                                scheduler=scheduler_pt,
                                num_epochs=args.pretrain_epoch,
                                device=device,
                                # early_stopping=early_stopping,
                                log_file=log_file)

        _, result = eval_model(mobilenet_pt, pt_dataloaders, dataset_sizes, device, log_file)

        print('> Saving Model...')
        print(' >> Saved Path: {}'.format(save_model_dir_path))
        save_model(model=mobilenet_pt,
                optimizer=optimizer_pt,
                file_path=os.path.join(save_model_dir_path,
                                'MobileNetV2_pt_ep{}_bs{}_lr{}.pth'.format(args.finetune_epoch,
                                                                            args.finetune_batch_size,
                                                                            args.finetune_learning_rate)))
        with open(os.path.join(save_model_dir_path, 'result_pt.txt'), 'w') as out:
            out.write(', '.join([str(x) for x in result])) 


    if args.finetune:
        if args.pretrain_model_path:
            # load ckpt
            print('> Loading ckpt: {}'.format(args.pretrain_model_path))
            checkpoint = torch.load(args.pretrain_model_path, weights_only=True)
            mobilenet_pt.load_state_dict(checkpoint['model_state_dict'])
            # redefine save path
            save_model_dir_path = os.path.join(os.path.dirname(args.pretrain_model_path), 'tag_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            os.mkdir(save_model_dir_path)
            # redefine log file path
            log_file = os.path.join(save_model_dir_path, "log.txt")

        # Finetuning
        optimizer_ft = optim.RMSprop(mobilenet.parameters(), lr=args.finetune_learning_rate)
        scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft,
                                                mode='max', 
                                                threshold=args.min_delta,
                                                patience=args.patience,
                                                factor=args.factor,
                                                cooldown=args.cooldown,
                                                min_lr=0.00000001)
        early_stopping = EarlyStopping(patience=10, min_delta=0.005)
        
        ft_train_dataloader = torch.utils.data.DataLoader(image_datasets['train'], 
                                                        batch_size=args.finetune_batch_size, 
                                                        shuffle=True, 
                                                        num_workers=4)
        ft_dataloaders = {'train': ft_train_dataloader, 'val': val_dataloader}
        custom_print('> Finetuning...', log_file)
        mobilenet_ft = train_model(model=mobilenet_pt,
                                dataloaders=ft_dataloaders,
                                dataset_sizes=dataset_sizes,
                                optimizer=optimizer_ft,
                                scheduler=scheduler_ft,
                                num_epochs=args.finetune_epoch,
                                device=device,
                                # early_stopping=early_stopping,
                                log_file=log_file)
        _, result = eval_model(mobilenet_ft, ft_dataloaders, dataset_sizes, device, log_file)

        custom_print('> Saving Model...', log_file)
        custom_print(' >> Saved Path: {}'.format(save_model_dir_path), log_file)
        save_model(model=mobilenet_ft,
                    optimizer=optimizer_ft,
                    file_path=os.path.join(save_model_dir_path,
                                'MobileNetV2_ft_ep{}_bs{}_lr{}.pth'.format(args.finetune_epoch,
                                                                            args.finetune_batch_size,
                                                                            args.finetune_learning_rate)))
        with open(os.path.join(save_model_dir_path, 'args.json'), 'w') as out:
            json.dump(vars(args), out)
        with open(os.path.join(save_model_dir_path, 'result_ft.txt'), 'w') as out:
            out.write(', '.join([str(x) for x in result])) 


    if args.evaluate:
        custom_print('> Evaluating Model...', log_file)
        if not args.model_path:
            args.model_path = save_model_dir_path
        # TODO: this is only true for continuing evaluation. consider for only evaluate case.
        # TODO: redefine log_file for only evaluate case.
        model_weight_path = os.path.join(args.model_path,
                                        'MobileNetV2_ft_ep{}_bs{}_lr{}.pth'.format(args.finetune_epoch,
                                                                                    args.finetune_batch_size,
                                                                                    args.finetune_learning_rate))

        custom_print(' >> Evaluating {}'.format(args.model_path), log_file)
        checkpoint = torch.load(model_weight_path, weights_only=True)
        mobilenet_pt.load_state_dict(checkpoint['model_state_dict'])
        _, result = eval_model(mobilenet_ft, {'val': val_dataloader}, dataset_sizes, device, log_file)
            
        # # mlflow log model
        # mlflow.pytorch.log_model(mobilenet_ft, "finetuned_model")

if __name__== '__main__':
    main()