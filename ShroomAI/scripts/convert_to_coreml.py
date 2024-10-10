import argparse
import coremltools as ct
import json
import os
import torch

import sys
sys.path.append('.')

from ShroomAI.models.shroomnet import ShroomNet

def main():
    parser = argparse.ArgumentParser(description='Converting Torch Checkpoint to MLpackage Checkpoint')

    parser.add_argument('--model_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/ckpt/mobilenet_v2_20241008_091337/mobilenet_v2_ft_ep30_bs256_lr1e-05.pth')
    parser.add_argument('--label_map_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/ckpt/mobilenet_v2_20241008_091337/label_map.json')
    parser.add_argument('--mlpackage_save_name', type=str, default='mobilenet_v2_finetuned.mlpackage')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--set_meta_data', action='store_true')
    
    args = parser.parse_args()

    mlpackage_save_path = os.path.join(os.path.dirname(args.model_path), args.mlpackage_save_name)
    
    # load model
    model = ShroomNet(num_classes=(args.num_classes, ), model_name='mobilenet_v2')
    checkpoint = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    example_input = torch.rand(1, 3, args.img_size, args.img_size) 
    traced_model = torch.jit.trace(model, example_input)

    # prepare class labels
    with open(args.label_map_path) as f:
        label_map = json.load(f)
    class_labels = list(label_map.keys())

    # convert the ckpt (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
    # https://medium.com/@kuluum/pytroch-to-coreml-cheatsheet-fda57979b3c6
    image_input = ct.ImageType(name='mobilenetv2_1.00_224_input', 
                            shape=example_input.shape,
                            scale = 1/(0.226*255.0),
                            bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)])
    classifier_config = ct.ClassifierConfig(class_labels)

    mlmodel = ct.converters.convert(traced_model, 
                                    source='pytorch',
                                    convert_to='mlprogram',
                                    inputs=[image_input], 
                                    classifier_config=classifier_config)

    # set metadata
    if args.set_meta_data:
        mlmodel.author = 'Seohyeong Jeong'
        mlmodel.short_description = 'Mushroom Image Classification (currently 1,000 species supported, mobilenet_v2_ft_ep30_bs256_lr1e-05)'
        mlmodel.version = '1.0.0'

    # save model
    mlmodel.save(mlpackage_save_path)
    
if __name__=='__main__':
    main()