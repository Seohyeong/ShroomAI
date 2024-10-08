import argparse
import datetime
import json
import os
import sys
sys.path.append('.')

import torch
import torch.backends.cudnn as cudnn

from ShroomAI.dataset.dataset import MushroomDataset
from ShroomAI.models.shroomnet import ShroomNet
from ShroomAI.train import prepare_and_train
from ShroomAI.utils.utils import custom_print

cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser(description='Training MobileNetV2 with GBIF Mushroom Dataset')

    # path
    parser.add_argument('--dataset_dir_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/dataset/inat_300')
    parser.add_argument('--meta_info_path', type=str,
                        default=None) # '/home/user/seohyeong/ShroomAI/ShroomAI/dataset/images_100_3314_meta.json'
    parser.add_argument('--ckpt_dir_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/ckpt')
    parser.add_argument('--pt_model_path', type=str,
                        default=None,
                        help='continue finetuning with the pretrained checkpoint')

    # backbone model
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', 
                        choices=['mobilenet_v2', 'efficientnet_b0', 'mobilevitv2-0.75'])
    
    # lr decay
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--cooldown', type=int, default=0)
    parser.add_argument('--min_delta', type=float, default=0.01)
    parser.add_argument('--factor', type=float, default=0.25)

    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--eval_bs', type=int, default=1024, help='evaluation batch size') # 1024

    # pretrain
    parser.add_argument('--pt_epoch', type=int, default=20, help='epoch for finetuning classification head') # 20
    parser.add_argument('--pt_bs', type=int, default=1024, help='training batch size')
    parser.add_argument('--pt_lr', type=float, default=0.0005) # 0.0005

    # finetune
    parser.add_argument('--ft_epoch', type=int, default=40, help='epoch for full finetuning') # 40
    parser.add_argument('--ft_bs', type=int, default=512, help='training batch size')
    parser.add_argument('--ft_lr', type=float, default=0.00001) # 0.0001

    # options
    parser.add_argument('--pretrain', action='store_true', help='classification head training')
    parser.add_argument('--finetune', action='store_true', help='full finetuning')

    args = parser.parse_args()

    if not args.pretrain and args.finetune:
        assert args.pt_model_path, "Pretrain first to finetune, otherwise pass pretrain_model_path."

    # Dataloaders
    print('\n> Building Dataloader...')
    train_dataset = MushroomDataset(os.path.join(args.dataset_dir_path, 'train'), args.meta_info_path, mode='train')
    val_dataset = MushroomDataset(os.path.join(args.dataset_dir_path, 'val'), args.meta_info_path, mode='val')
    assert train_dataset.num_classes == val_dataset.num_classes
    num_classes = train_dataset.num_classes # tuple
    print(' >> # class: {}, # train samples: {}, # val samples: {}'.format(
        num_classes, len(train_dataset), len(val_dataset)))

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.pretrain:
        
        save_dir = os.path.join(args.ckpt_dir_path, '{}_{}'.format(args.model_name, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        os.mkdir(save_dir)
        log_file = os.path.join(save_dir, "log.txt")
        
        # save label map
        label_map_path = os.path.join(save_dir, "label_map.json")
        with open(label_map_path, 'w') as outfile: 
            json.dump(train_dataset.name2label, outfile, indent=4)
        
        custom_print(f'\n> Loading Pre-trained {args.model_name}...', log_file)
        shroomnet = ShroomNet(num_classes=num_classes, model_name=args.model_name)
        total_params = sum(p.numel() for p in shroomnet.parameters() if p.requires_grad)
        custom_print(f' >> Total # of Trainable Params: {total_params:,}...', log_file)
        
        shroomnet = shroomnet.to(device)

        trained_model = prepare_and_train(args, shroomnet, train_dataset, val_dataset, device, log_file, save_dir, phase='pretrain')


    if args.finetune:
        
        if args.pt_model_path:
            # load ckpt
            print('\n> Loading ckpt: {}'.format(args.pt_model_path))
            checkpoint = torch.load(args.pt_model_path, weights_only=True)
            shroomnet = ShroomNet(num_classes=num_classes, model_name=args.model_name)
            shroomnet.load_state_dict(checkpoint['model_state_dict'])
            trained_model = shroomnet.to(device)
            # redefine save path
            save_dir = os.path.join(os.path.dirname(args.pt_model_path), 'tag_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            os.mkdir(save_dir)
            # redefine log file path
            log_file = os.path.join(save_dir, "log.txt")

        _ = prepare_and_train(args, trained_model, train_dataset, val_dataset, device, log_file, save_dir, phase='finetune')


if __name__== '__main__':
    main()