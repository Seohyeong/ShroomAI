import os
import shutil
import datetime
from tqdm import tqdm

# create another images_100/train directory structure with images
max_img_per_class = 100

image_dir_path = '/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/images/train'
new_image_dir_path = '/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/images_100_{}/train'.format(datetime.datetime.now().strftime('%Y%m%d'))

os.makedirs(new_image_dir_path)

for label in tqdm(os.listdir(image_dir_path), total = len(os.listdir(image_dir_path))):

    if label.startswith('.'):
        continue
    
    current_label_dir_path = os.path.join(image_dir_path, label)
    new_label_dir_path = os.path.join(new_image_dir_path, label)
    if not os.path.exists(new_label_dir_path):
        os.mkdir(new_label_dir_path)
    
    count = 0
    for item in os.listdir(current_label_dir_path):
        current_item_path = os.path.join(current_label_dir_path, item)
        new_item_path = os.path.join(new_label_dir_path, item)
        shutil.move(current_item_path, new_item_path)
        count += 1

        if count >= max_img_per_class:
            break