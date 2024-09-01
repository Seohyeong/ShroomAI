import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
import random
import shutil
import zipfile

import tensorflow as tf
import tensorflow_datasets as tfds


def get_result(history, result=dict()):
    if len(result) > 0:
        result['finetune_full_acc'] = history.history['accuracy']
        result['finetune_full_val_acc'] = history.history['val_accuracy']
        result['finetune_full_loss'] = history.history['loss']
        result['finetune_full_val_loss'] = history.history['val_loss']
    else:
        result['finetune_head_acc'] = history.history['accuracy']
        result['finetune_head_val_acc'] = history.history['val_accuracy']
        result['finetune_head_loss'] = history.history['loss']
        result['finetune_head_val_loss'] = history.history['val_loss']
    return result


def get_top_n_acc(validation, model):
    count = 0
    top1, top2, top3, top4, top5 = 0, 0, 0, 0, 0

    for img, label in tqdm(validation.take(len(validation)), total=len(validation)):
        label = int(label)
        # pred_logits = model.predict(tf.expand_dims(img, axis=0), verbose=0)[0]
        pred_logits = model.predict(img, verbose=0)[0]
        preds = pred_logits.argsort()[-5:][::-1]
        
        if label in preds:
            top5 += 1
        if label in preds[:4]:
            top4 += 1
        if label in preds[:3]:
            top3 += 1
        if label in preds[:2]:
            top2 += 1
        if label == preds[0]:
            top1 += 1

        count += 1
    return {
        'top_1_acc': top1/count,
        'top_2_acc': top2/count,
        'top_3_acc': top3/count,
        'top_4_acc': top4/count,
        'top_5_acc': top5/count
    }


def main():
    parser = argparse.ArgumentParser(description='Training MobileNetV2 with GBIF Mushroom Dataset')

    parser.add_argument('--dataset_dir_path', type=str, 
                        default='/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/images/')
    parser.add_argument('--ckpt_dir_path', type=str, default='/content/drive/MyDrive/fungi/ckpt')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='pass saved model path to run evaluation')
    
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--finetuning_learning_rate', type=float, default=0.0001)
    parser.add_argument('--epoch_classification_head', type=int, default=20, help='epoch for finetuning classification head')
    parser.add_argument('--epoch', type=int, default=20, help='epoch for full finetuning')
    parser.add_argument('--finetune_at', type=int, default=100)
    parser.add_argument('--batch_shuffle_buffer_size', type=int, default=1000)

    parser.add_argument('--use_google_colab', action='store_true')
    parser.add_argument('--finetune', action='store_true', 
                        help='full finetuning')
    parser.add_argument('--evaluate', action='store_true')

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    args.use_google_colab = True
    args.finetune = True
    args.evaluate = True

    start_time = time.time()

    if args.use_google_colab:
        print('\n> Setting Up Google Colab...')
        # from google.colab import drive
        # drive.mount('/content/drive')

        if not os.path.exists('./images_100_combined'):
            path_to_zip_file = '/content/drive/MyDrive/fungi/images_100_combined.zip'
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall('.')
        args.dataset_dir_path = './images_100_combined/'


    if not os.path.exists(args.ckpt_dir_path):
        os.mkdir(args.ckpt_dir_path)


    # Dataloader
    print('\n> Building Dataloader...')
    builder = tfds.folder_dataset.ImageFolder(args.dataset_dir_path)
    num_classes = builder.info.features["label"].num_classes
    num_train_examples = builder.info.splits['train'].num_examples
    num_test_examples = builder.info.splits['test'].num_examples
    print(' >> # class: {}, # train samples: {}, # val samples: {}'.format(
        num_classes, num_train_examples, num_test_examples))

    raw_train = builder.as_dataset(split='train', shuffle_files=True)
    raw_valid = builder.as_dataset(split='test', shuffle_files=True)

    def format_example(pair):
        image, label = pair['image'], pair['label']
        image = tf.cast(image, tf.float32)
        image = (image/127.5) - 1 # TODO: [0, 1] or [-1, 1]
        image = tf.image.resize(image, (args.img_size, args.img_size))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=random.sample([1,2,3,4], 1)[0])
        return image, label

    train = raw_train.map(format_example)
    validation = raw_valid.map(format_example)

    train_batches = train.shuffle(args.batch_shuffle_buffer_size).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    validation_batches = validation.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    if args.finetune:
        # Load Pre-trained MobileNet V2
        print('\n> Loading Pre-trained MobileNetV2...')
        base_model = tf.keras.applications.MobileNetV2(input_shape=(args.img_size, args.img_size, 3),
                                                    include_top=False,
                                                    weights='imagenet')

        model = tf.keras.Sequential()
        model.add(base_model)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        # step 1
        base_model.trainable = False
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                        mode='max',
                                                        min_delta=0.01,
                                                        patience=1,
                                                        factor=0.25,
                                                        verbose=1,
                                                        cooldown=0,
                                                        min_lr=0.00000001)
        
        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         mode='max',
                                                         min_delta=0.005,
                                                         patience=10,
                                                         verbose=1,
                                                         restore_best_weights=True)
        
        print('\n> Training Classification Head...')
        history = model.fit(train_batches,
                            epochs=args.epoch_classification_head,
                            validation_data=validation_batches,
                            callbacks=[early_stopper, reduce_lr])
        print(' >> Training Completed!')
        result = get_result(history)

        # step 2
        base_model.trainable = True
        for layer in base_model.layers[:args.finetune_at]:
            layer.trainable =  False

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.finetuning_learning_rate),
                    metrics=['accuracy'])

        print('\n> Finetuning...')
        history_finetune = model.fit(train_batches,
                                 epochs=args.epoch_classification_head + args.epoch,
                                 initial_epoch = args.epoch_classification_head,
                                 validation_data=validation_batches,
                                 callbacks=[early_stopper, reduce_lr])
        print(' >> Finetuning Completed!')

        result = get_result(history_finetune, result)

        # save
        print('\n> Saving Model...')
        save_model_dir_path = os.path.join(args.ckpt_dir_path, 'model_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        os.mkdir(save_model_dir_path)
        print(' >> Saved Path: {}'.format(save_model_dir_path))
        model.save(os.path.join(save_model_dir_path, 'model_weight'))
        with open(os.path.join(save_model_dir_path, 'args.json'), 'w') as out:
            json.dump(vars(args), out)
        with open(os.path.join(save_model_dir_path, 'result.json'), 'w') as out:
            json.dump(result, out)
        label_map = builder.info.features["label"]._str2int
        with open(os.path.join(save_model_dir_path, 'label_map.json'), 'w') as out:
            json.dump(label_map, out)
        print(' >> Saving Completed!')


    if args.evaluate:
        print('\n> Evaluating Model...')
        if not args.model_path:
            args.model_path = save_model_dir_path
        model_weight_path = os.path.join(args.model_path, 'model_weight')

        model = tf.keras.models.load_model(model_weight_path)
        test_batches = validation.batch(1)
        top_n_result = get_top_n_acc(test_batches, model)
        print(top_n_result)
        print(' >> top1: {}\n    top2:{}\n    top3:{}\n    top4:{}\n    top5:{}'.format(
            top_n_result['top_1_acc'], 
            top_n_result['top_2_acc'], 
            top_n_result['top_3_acc'], 
            top_n_result['top_4_acc'], 
            top_n_result['top_5_acc'],
        ))

        with open(os.path.join(args.model_path, 'top_5_acc.json'), mode='w') as out:
            json.dump(top_n_result, out)

    elapsed_time = time.time() - start_time
    print('\n> Total Elaspsed Time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))