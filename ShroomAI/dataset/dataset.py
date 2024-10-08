import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MushroomDataset(Dataset):
    def __init__(self, root_dir, meta_info_path=None, mode='train'):
        self.root_dir = root_dir # /train, /test
        self.meta_info_path = meta_info_path
        self.mode = mode
        
        self.transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        self.class_names = os.listdir(root_dir)
        
        self.image_paths = []
        self.labels = []
        self.name2label = {} # class name to idx
        self.label2name = {} # idx to class name
        
        for label, class_name in enumerate(self.class_names):
            class_folder = os.path.join(root_dir, class_name)
            self.name2label[class_name] = label
            self.label2name[label] = class_name
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)
            
        self.num_classes = (len(self.class_names),)
        
        # to get str genus: a = self.name2genus[self.label2name[self.labels[idx]]]
        # to get idx genus: self.genus_labels.index(a)
        if meta_info_path:
            self.name2genus = {}
            self.genus_labels = set()
            with open(meta_info_path, 'r') as file:
                meta_info_json = json.load(file)
            for info in meta_info_json:
                self.name2genus[info['species']] = info['genus']
                self.genus_labels.add(info['genus'])
            self.genus_labels = list(sorted(self.genus_labels))
            
            self.num_classes = (len(self.class_names), len(self.genus_labels))
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        species_label = self.labels[idx]
        label = (species_label,)
        
        if self.mode in self.transforms:
            image = self.transforms[self.mode](image)
            
        if self.meta_info_path:
            genus = self.name2genus[self.label2name[self.labels[idx]]]
            genus_label = self.genus_labels.index(genus)
            label = (species_label, genus_label)
        
        return image, label