import argparse
import yaml

import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.append('mlcvnets')
# from mlcvnets.cvnets.models.classification.mobilevit_v2 import MobileViTv2
# from mlcvnets.options.opts import get_training_arguments
# from mlcvnets.cvnets import get_model
# from mlcvnets.tests.configs import get_config
# from mlcvnets.utils.common_utils import device_setup

class ShroomNet(nn.Module):
    def __init__(self, num_classes, model_name):
        super(ShroomNet, self).__init__()
        
        self.species_num_classes, *self.genus_num_classes = num_classes
        
        if model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights='IMAGENET1K_V1') # 3,504,872
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1') # 5,288,548
        # elif model_name == 'mobilevitv2-0.75':
        #     # TODO: sort out path problem
        #     config_file_path = "tests/data/datasets/classification/dummy_configs/image_classification_dataset.yaml"
        #     opts = get_config(config_file=config_file_path)
        #     # device set-up
        #     opts = device_setup(opts)
        #     with open('/home/user/seohyeong/ShroomAI/ShroomAI/models/mobilevitv2-0.75.yaml', 'r') as file:
        #         data = yaml.safe_load(file)
        #         opts = argparse.Namespace(**data)
        #     self.model = get_model(opts)
        #     # self.model = MobileViTv2('mobilevitv2-0.75.yaml')
        #     # self.model = torch.load('mobilevitv2-0.75.pt')
            
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
        )
        
        self.species_head = nn.Linear(in_features, self.species_num_classes)
        if self.genus_num_classes:
            self.genus_head = nn.Linear(in_features, self.genus_num_classes[0])
        
    def forward(self, x):
        feature = self.model(x)
        out_species = self.species_head(feature)
        out = (out_species, )
        if self.genus_num_classes:
            out_genus = self.genus_head(feature)
            out = (out_species, out_genus)
        return out
        