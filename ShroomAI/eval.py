import argparse
import os

import torch
import torch.nn as nn

from ShroomAI.dataset.dataset import MushroomDataset
from ShroomAI.models.shroomnet import ShroomNet
from ShroomAI.utils.utils import custom_print

# TODO: merge this function with train() in train.py
def evaluate(model, dataloader, dataset_size, device, log_file):
    model.eval()

    running_loss = 0.0
    running_corrects_topn = [0] * 5

    with torch.set_grad_enabled(False):
        for inputs, labels in dataloader:
            labels_species, *labels_genus = labels
            
            inputs = inputs.to(device)
            labels_species = labels_species.to(device)
            if labels_genus:
                labels_genus = labels_genus[0].to(device)

            outputs_species, *outputs_genus = model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs_species, labels_species)
            
            _, topk_indices = torch.topk(outputs_species, 5, dim=1)

            for k in range(1, 6):
                correct_topk = topk_indices[:, :k].eq(labels_species.view(-1, 1).expand_as(topk_indices[:, :k]))
                running_corrects_topn[k-1] += correct_topk.sum().item()

            running_loss += loss.item() * inputs.size(0)

    loss = running_loss / dataset_size
    topn_accuracies = [corrects / dataset_size for corrects in running_corrects_topn]
    
    formatted_strs = [' >> top {}: {:.4f}    '.format(n + 1, acc) for n, acc in enumerate(topn_accuracies)]
    custom_print(''.join(formatted_strs), log_file)


def main():
    parser = argparse.ArgumentParser(description='Evaluation Script on GBIF Mushroom Dataset')
    parser.add_argument('--dataset_dir_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/dataset/images_100_3314')
    parser.add_argument('--eval_model_path', type=str, 
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/ckpt/model_20240922_101333/MobileNetV2_pt_ep40_bs128_lr1e-05.pth')
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    args = parser.parse_args()

    # Log file path
    log_file = os.path.join(os.path.dirname(args.eval_model_path), 'eval_log_file.txt')
    
    # Dataloader
    print('\n> Building Dataloader...')
    val_dataset = MushroomDataset(os.path.join(args.dataset_dir_path, 'val'), mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=args.eval_batch_size, 
                                                 shuffle=False, 
                                                 num_workers=4)

    print(' >> # class: {}, # val samples: {}'.format(val_dataset.num_classes, len(val_dataset)))
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Model
    shroomnet = ShroomNet(num_classes=val_dataset.num_classes)
    shroomnet.to(device)

    custom_print('\n> Evaluating Model...', log_file)
    custom_print(' >> Evaluating {}'.format(args.eval_model_path), log_file)
    
    checkpoint = torch.load(args.eval_model_path, weights_only=True)
    shroomnet.load_state_dict(checkpoint['model_state_dict'])
    evaluate(shroomnet, val_dataloader, len(val_dataset), device, log_file)


if __name__== '__main__':
    main()