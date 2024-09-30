import torch.nn as nn
from torchvision import models

class ShroomNet(nn.Module):
    def __init__(self, num_classes, model_name):
        super(ShroomNet, self).__init__()
        
        self.species_num_classes, *self.genus_num_classes = num_classes
        
        if model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights='IMAGENET1K_V1') # 3,504,872
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1') # 5,288,548
            
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
        