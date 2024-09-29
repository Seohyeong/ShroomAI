import torch.nn as nn
from torchvision import models

class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        
        self.species_num_classes, *self.genus_num_classes = num_classes
        
        self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        in_features = self.mobilenet.classifier[1].in_features
        
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(1024, num_classes)
        )
        
        self.species_head = nn.Linear(1024, self.species_num_classes)
        if self.genus_num_classes:
            self.genus_head = nn.Linear(1024, self.genus_num_classes[0])
        
    def forward(self, x):
        feature = self.mobilenet(x)
        out_species = self.species_head(feature)
        out = (out_species, )
        if self.genus_num_classes:
            out_genus = self.genus_head(feature)
            out = (out_species, out_genus)
        return out
        