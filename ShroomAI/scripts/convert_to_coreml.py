import torch
import coremltools as ct
import json

import sys
sys.path.append('.')

from ShroomAI.models.shroomnet import ShroomNet

model_path = '/home/user/seohyeong/ShroomAI/ShroomAI/ckpt/mobilenet_v2_20241008_091337/mobilenet_v2_ft_ep30_bs256_lr1e-05.pth'
label_map_path = '/home/user/seohyeong/ShroomAI/ShroomAI/ckpt/mobilenet_v2_20241008_091337/label_map.json'

model = ShroomNet(num_classes=(1000, ), model_name='mobilenet_v2')
checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

example_input = torch.rand(1, 3, 224, 224) 
traced_model = torch.jit.trace(model, example_input)

with open(label_map_path) as f:
    label_map = json.load(f)

class_labels = list(label_map.keys())

# set config
image_input = ct.ImageType(name='mobilenetv2_1.00_224_input', 
                           shape=example_input.shape, # (1, 224, 224, 3,) 
                           bias=[- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)], 
                           scale=1/(0.226*255.0))
classifier_config = ct.ClassifierConfig(class_labels)

# convert
mlmodel = ct.converters.convert(traced_model, 
                                source='pytorch',
                                convert_to='mlprogram',
                                inputs=[image_input], 
                                classifier_config=classifier_config)

# set metadata
mlmodel.author = 'Seohyeong Jeong'
mlmodel.short_description = 'Mushroom Image Classification (currently 1,000 species supported, mobilenet_v2_ft_ep30_bs256_lr1e-05)'
mlmodel.version = '1.0.0'

# save mlmodel
mlmodel.save('/home/user/seohyeong/ShroomAI/ShroomAI/ckpt/mobilenet_v2_20241008_091337/mobilenet_v2.mlpackage')