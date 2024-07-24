import argparse
from collections import defaultdict
import concurrent.futures
import datetime
from io import BytesIO
import math
import numpy as np
import os
import pandas as pd
from PIL import Image
from random import sample
import requests
import shutil
from tqdm import tqdm


def get_image(url, save_path, resize_size, run_parallel=False, print_error=False):
    try:
        response = requests.get(url)
        response.raise_for_status()
        try:
            image = Image.open(BytesIO(response.content))
            resized_image = image.resize((resize_size, resize_size))
            resized_image.save(save_path)
            if run_parallel:
                return True
        except Exception as e:
            if print_error:
                print(f"    >> Failed to LOAD image: {e}")
            if run_parallel:
                return False
    except requests.exceptions.RequestException as e:
        if print_error:
            print(f"    >> Failed to DOWNLOAD image: {e}")
        if run_parallel:
            return False
    

def get_urls_and_save_paths(filtered_df, args, split='train'):
    if args.exclude:
        exclude_path = os.path.join(args.dataset_dir_path, args.exclude)

    image_urls = []
    save_paths = []
    class_to_paths = defaultdict(list)

    filtered_df = filtered_df.sample(len(filtered_df))

    print("> Getting image urls and save paths...")

    for _, row in tqdm(filtered_df.iterrows(), total = len(filtered_df)):
        url = str(row["identifier"])
        label = str.lower(row["species"]).replace(" ", "_")
        label_dir_path = os.path.join(args.img_dir_path, split, label)
        save_path = os.path.join(label_dir_path, str(row["gbifID"])+ ".jpg")

        if len(os.listdir(label_dir_path)) + len(class_to_paths[label]) >= args.max_images_per_class:
            continue
        
        if args.exclude:
            exclude_train_path = os.path.join(exclude_path, 'train', label, str(row["gbifID"])+ ".jpg")
            exclude_test_path = os.path.join(exclude_path, 'test', label, str(row["gbifID"])+ ".jpg")
            if os.path.exists(exclude_train_path) or os.path.exists(exclude_test_path):
                continue

        if os.path.exists(save_path):
            continue
        else:
            image_urls.append(url)
            save_paths.append(save_path)
            class_to_paths[label].append(row['gbifID'])

    return image_urls, save_paths


def run(filtered_df, args):
    image_urls, save_paths = get_urls_and_save_paths(filtered_df, args)

    if args.run_parallel:
        print('> Downloading Images in Parallel...')
        def download_with_progress(url, save_path):
            success = get_image(url, save_path, args.resize_size, run_parallel=True, print_error=args.print_error)
            if success:
                pbar.update(1)

        with tqdm(total=len(image_urls)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                for url, save_path in zip(image_urls, save_paths):
                    futures.append(executor.submit(download_with_progress, url, save_path))

                for future in concurrent.futures.as_completed(futures):
                    future.result() 

        pbar.close()

    else:
        print('> Downloading Images...')
        for url, save_path in tqdm(zip(image_urls, save_paths), total=len(image_urls)):
            get_image(url, save_path, args.resize_size, run_parallel=False, print_error=args.print_error)


def print_stat(args, split):
    class_to_num = dict()
    for folder_name in os.listdir(os.path.join(args.img_dir_path, split)):
        if folder_name == '.DS_Store':
            continue
        folder_path = os.path.join(args.img_dir_path, split, folder_name)
        num_files = len(os.listdir(folder_path))
        class_to_num[folder_name] = num_files
    print('\n**** Statistics for {} (# of classes: {}) ****'.format(split, len(class_to_num)))
    print(class_to_num)


def main():
    parser = argparse.ArgumentParser(description='GBIF Mushroom Dataset Image Scraping')

    parser.add_argument('--dataset_dir_path', type=str, 
                        default='/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset')
    parser.add_argument('--img_dir_name', type=str, default='images_100_20240724')
    parser.add_argument('--exclude', type=str, default='images_100_combined')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=12)
    parser.add_argument('--max_images_per_class', type=int, default=100)
    parser.add_argument('--resize_size', type=int, default=224)

    parser.add_argument('--split_train_test', action='store_true', help='whether to split downloaded images into train/test split')
    parser.add_argument('--train_test_ratio', type=float, default=0.9, help='if 0.8 then 8:2 = train:test')

    parser.add_argument('--run_parallel', action='store_true', help='run in parallel')
    parser.add_argument('--print_error', action='store_true', help='print image download related errors')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # create image folder structure 
    if not args.img_dir_name:
        args.img_dir_name = 'images_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        args.img_dir_path = os.path.join(args.dataset_dir_path, args.img_dir_name)
        if not os.path.exists(args.img_dir_path):
            os.mkdir(args.img_dir_path)
            os.mkdir(os.path.join(args.img_dir_path, 'train'))
            os.mkdir(os.path.join(args.img_dir_path, 'test'))
        
        print('> Downloading Images to: {}'.format(args.img_dir_path))
        print(' > Creating Directory: {}'.format(os.path.join(args.img_dir_path, 'train')))
        print(' > Creating Directory: {}'.format(os.path.join(args.img_dir_path, 'test')))

        filtered_df = pd.read_csv(os.path.join(args.dataset_dir_path, 'sampled_df.txt'), sep='\t')
        
        # create label folders inside image folder structure
        print('     > Creating Subdirectory of label...')
        for species in filtered_df['species'].unique():
            label = str.lower(species).replace(" ", "_")
            train_label_dir_path = os.path.join(args.img_dir_path, 'train', label)
            test_label_dir_path = os.path.join(args.img_dir_path, 'test', label)
            if not os.path.exists(train_label_dir_path):
                os.mkdir(train_label_dir_path)
            if not os.path.exists(test_label_dir_path):
                os.mkdir(test_label_dir_path)
    else:
        filtered_df = pd.read_csv(os.path.join(args.dataset_dir_path, 'sampled_df.txt'), sep='\t')
        args.img_dir_path = os.path.join(args.dataset_dir_path, args.img_dir_name)

    # first download images to 'train' folder
    run(filtered_df, args)
    print('     > Downloading Images Completed!')

    # then seperate to train/test
    if args.split_train_test:
        print('> Spliting Images to train/test...')
        for label in tqdm(os.listdir(os.path.join(args.img_dir_path, 'train')), 
                        total=len(os.listdir(os.path.join(args.img_dir_path, 'train')))):
            if label.startswith('.'): # .DS_Store
                continue

            if not os.path.exists(os.path.join(args.img_dir_path, 'test', label)):
                    os.makedirs(os.path.join(args.img_dir_path, 'test', label))

            samples = os.listdir(os.path.join(args.img_dir_path, 'train', label))
            num_test_samples = max(math.ceil(len(samples) - args.train_test_ratio * args.max_images_per_class), 0)
            test_samples = sample(samples, num_test_samples)

            for test_sample in test_samples:
                current_path = os.path.join(args.img_dir_path, 'train', label, test_sample)
                new_path = os.path.join(args.img_dir_path, 'test', label, test_sample)
                shutil.move(current_path, new_path)

        # print stat
        print('> Printing Statistics for train/test...')
        print_stat(args, split='train')
        print_stat(args, split='test')
    else:
        # print stat
        print('> Printing Statistics for train/test...')
        print_stat(args, split='train')


if __name__ == '__main__':
    main()