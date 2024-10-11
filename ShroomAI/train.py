import copy
import json
import os
from tempfile import TemporaryDirectory
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from ShroomAI.eval import evaluate
from ShroomAI.utils.utils import EarlyStopping, custom_print, save_model

cudnn.benchmark = True


def train(model, dataloaders, dataset_sizes, optimizer, scheduler, num_epochs, device, 
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
                running_loss_species = 0.0
                running_loss_genus = 0.0
                running_corrects = 0
                
                flag_multitask = False

                # Iterate over data.
                if phase == 'train':
                    dataloader = tqdm(dataloaders[phase], 
                                    total=len(dataloaders[phase]), 
                                    desc=f' >> [{phase}] Epoch {epoch+1}/{num_epochs}')
                else:
                    dataloader = dataloaders[phase]
                    
                for inputs, labels in dataloader:
                    labels_species, *labels_genus = labels
                    
                    inputs = inputs.to(device)
                    labels_species = labels_species.to(device)
                    if labels_genus:
                        labels_genus = labels_genus[0].to(device)

                    model.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs_species, *outputs_genus = model(inputs) # TODO: model needs to output output2
                        _, preds_species = torch.max(outputs_species, 1)
                        criterion = nn.CrossEntropyLoss()
                        loss_species = criterion(outputs_species, labels_species)
                        loss_genus = 0 # TODO: test
                        if outputs_genus:
                            flag_multitask = True
                            outputs_genus = outputs_genus[0]
                            _, preds_genus = torch.max(outputs_genus, 1)
                            criterion = nn.CrossEntropyLoss()
                            loss_genus = criterion(outputs_genus, labels_genus)
                            
                        loss = loss_species + loss_genus

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds_species == labels_species.data)
                    if flag_multitask:
                        running_loss_species += loss_species.item() * inputs.size(0)
                        running_loss_genus += loss_genus.item() * inputs.size(0)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                if flag_multitask:
                    epoch_loss_species = running_loss_species / dataset_sizes[phase]
                    epoch_loss_genus = running_loss_genus / dataset_sizes[phase]
                else:
                    epoch_loss_species = epoch_loss
                    epoch_loss_genus = 0.0

                if phase == 'val':
                    scheduler.step(epoch_loss)
                    if early_stopping:
                        early_stopping(epoch_loss)
                        if early_stopping.early_stop:
                            custom_print(" >> Early stopping!", log_file)
                            model.load_state_dict(best_model_wts)
                            return model
                # TODO: log both species loss and genus loss
                log_msg = (
                    f' >> [Epoch {epoch + 1}/{num_epochs}] '
                    f'  {"Train" if phase == "train" else "Val"} Loss: {epoch_loss:.4f}'
                    f'  Species Loss: {epoch_loss_species:.4f}  Genus Loss: {epoch_loss_genus:.4f}'
                    f'  {"Train" if phase == "train" else "Val"} Acc: {epoch_acc:.4f}'
                    f'  {"Current lr: " + str(optimizer.param_groups[0]["lr"]) if phase == "train" else ""}'
                )
                custom_print(log_msg, log_file)
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_model_params_path)
        
        time_elapsed = time.time() - since
        custom_print(f' >> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Best val Acc: {best_acc:4f}',
                     log_file)

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


def prepare_and_train(args, model, train_dataset, val_dataset, device, log_file, save_dir, run_eval=True, phase=None):
    # Prepare
    if phase == 'pretrain':
        for np, p in model.named_parameters():
            if ('classifier' in np) or ('head' in np):
                p.requires_grad = True
            else:
                p.requires_grad = False
                
        epoch = args.pt_epoch
        lr = args.pt_lr
        bs = args.pt_bs
        train_msg = '\n> Training Classification Head...'
        ckpt_name = '{}_pt_ep{}_bs{}_lr{}.pth'.format(args.model_name, args.pt_epoch, args.pt_bs, args.pt_lr)
    elif phase == 'finetune':
        for _, p in model.named_parameters():
            p.requires_grad = True
            
        epoch = args.ft_epoch
        lr = args.ft_lr
        bs = args.ft_bs
        train_msg = '\n> Finetuning...'
        ckpt_name = '{}_ft_ep{}_bs{}_lr{}.pth'.format(args.model_name, args.ft_epoch, args.ft_bs, args.ft_lr)
    
    if args.model_name == 'mobilenet_v2':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif args.model_name == 'efficientnet_b0':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                            mode='max', 
                                            threshold=args.min_delta,
                                            patience=args.patience,
                                            factor=args.factor,
                                            cooldown=args.cooldown,
                                            min_lr=0.00000001)
    early_stopping = EarlyStopping(patience=5)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=4)

    # Train
    custom_print(train_msg, log_file)
    trained_model = train(model=model,
                          dataloaders={'train': train_dataloader, 'val': val_dataloader},
                          dataset_sizes={'train': len(train_dataset), 'val': len(val_dataset)},
                          optimizer=optimizer,
                          scheduler=scheduler,
                          num_epochs=epoch,
                          device=device,
                          early_stopping=early_stopping,
                          log_file=log_file)

    # Eval
    if run_eval:
        evaluate(trained_model, val_dataloader, len(val_dataset), device, log_file)
    
    # Save
    custom_print('\n> Saving Model...', log_file)
    custom_print(' >> Saved Path: {}'.format(save_dir), log_file)
    
    save_model(model=trained_model, optimizer=optimizer, file_path=os.path.join(save_dir, ckpt_name))
    
    with open(os.path.join(save_dir, 'args.json'), 'w') as out:
        json.dump(vars(args), out, indent=4)
        
    return trained_model