import os
import shutil


img_dir_to_combine = [
    '/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/images_accum',
    '/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/images'
    ]
base_dir = img_dir_to_combine[0]
img_dir_to_combine = img_dir_to_combine[1:]

def combine(base_dir, dir, split):
    for label in os.listdir(os.path.join(dir, split)):
        if label.startswith('.'):
            continue
        label_dir = os.path.join(dir, split, label)
        new_label_dir = os.path.join(base_dir, split, label)
        if not os.path.exists(new_label_dir):
            os.mkdir(new_label_dir)
        for item in os.listdir(label_dir):
            current_path = os.path.join(label_dir, item)
            new_path = os.path.join(base_dir, split, label, item)
            if not os.path.exists(new_path):
                shutil.move(current_path, new_path)


def print_stat(img_dir_path, split):
    class_to_num = dict()
    for folder_name in os.listdir(os.path.join(img_dir_path, split)):
        if folder_name == '.DS_Store':
            continue
        folder_path = os.path.join(img_dir_path, split, folder_name)
        num_files = len(os.listdir(folder_path))
        class_to_num[folder_name] = num_files
    print('\nStat for {} [num classes: {}]'.format(split, len(class_to_num)))
    print(class_to_num)


for dir in img_dir_to_combine:
    combine(base_dir, dir, 'train')
    # combine(base_dir, dir, 'test')
print_stat(base_dir, 'train')
print_stat(base_dir, 'test')