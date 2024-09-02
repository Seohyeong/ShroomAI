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

import mlflow

from utils import get_result, get_top_n_acc, plot

np.random.seed(42)
tf.random.set_seed(42)


def main():
    parser = argparse.ArgumentParser(description='Training MobileNetV2 with GBIF Mushroom Dataset')

    # path
    parser.add_argument('--dataset_dir_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/dataset/images_100_3314')
    parser.add_argument('--ckpt_dir_path', type=str,
                        default='/home/user/seohyeong/ShroomAI/ShroomAI/ckpt')
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='pass saved model path to run evaluation')
    parser.add_argument('--pretrain_model_path', type=str,
                        default=None,
                        help='continue finetuning with the pretrained checkpoint')

    # lr decay
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--cooldown', type=int, default=0)
    parser.add_argument('--min_delta', type=float, default=0.01)
    parser.add_argument('--factor', type=float, default=0.25)

    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='evaluation batch size')
    parser.add_argument('--batch_shuffle_buffer_size', type=int, default=1000)

    # pretrain
    parser.add_argument('--pretrain_epoch', type=int, default=20, help='epoch for finetuning classification head') # 20
    parser.add_argument('--pretrain_batch_size', type=int, default=1024, help='training batch size')
    parser.add_argument('--pretrain_learning_rate', type=float, default=0.001) # 0.0005

    # finetune
    parser.add_argument('--finetune_at', type=int, default=0) # 100
    parser.add_argument('--finetune_epoch', type=int, default=40, help='epoch for full finetuning') # 40
    parser.add_argument('--finetune_batch_size', type=int, default=512, help='training batch size')
    parser.add_argument('--finetune_learning_rate', type=float, default=0.00005) # 0.0001

    # options
    parser.add_argument('--use_google_colab', action='store_true')
    parser.add_argument('--pretrain', action='store_true', help='classification head training')
    parser.add_argument('--finetune', action='store_true', help='full finetuning')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()
    # args, unknown = parser.parse_known_args() 

    args.use_google_colab = True
    args.pretrain = True
    args.finetune = True
    args.evaluate = True
    args.plot = False

    try:
        assert(tf.__version__=='2.13.0')
    except AssertionError as e:
        print("Tensorflow version must be 2.13.0. Current version: {}".format(tf.__version__))

    start_time = time.time()

    if not args.pretrain and args.finetune:
        assert args.pretrain_model_path, "Pretrain first to finetune, otherwise pass pretrain_model_path."

    if args.use_google_colab:
        print('\n> Setting Up Google Colab...')
        # from google.colab import drive
        # drive.mount('/content/drive')

        if not os.path.exists('./images'):
            path_to_zip_file = '/content/drive/MyDrive/fungi/images.zip'
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall('.')
        args.dataset_dir_path = './images/'

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

    def format_example_train(pair):
        image, label = pair['image'], pair['label']
        image = tf.cast(image, tf.float32)
        image = (image/127.5) - 1 # [-1, 1]
        image = tf.image.resize(image, (args.img_size, args.img_size))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=random.sample([1,2,3,4], 1)[0])
        return image, label

    def format_example_test(pair):
        image, label = pair['image'], pair['label']
        image = tf.cast(image, tf.float32)
        image = (image/127.5) - 1
        image = tf.image.resize(image, (args.img_size, args.img_size))
        return image, label

    train = raw_train.map(format_example_train)
    validation = raw_valid.map(format_example_test)

    validation_batches = validation.batch(args.eval_batch_size).prefetch(tf.data.AUTOTUNE)

    if args.pretrain:
        # Load Pre-trained MobileNet V2
        train_batches = train.shuffle(args.batch_shuffle_buffer_size).batch(args.pretrain_batch_size).prefetch(tf.data.AUTOTUNE)

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

        base_model.trainable = False
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.pretrain_learning_rate),
                      metrics=['accuracy'])

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                        mode='max',
                                                        min_delta=args.min_delta,
                                                        patience=args.patience,
                                                        factor=args.factor,
                                                        verbose=1,
                                                        cooldown=args.cooldown,
                                                        min_lr=0.00000001)

        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                         mode='max',
                                                         min_delta=0.005,
                                                         patience=10,
                                                         verbose=1,
                                                         restore_best_weights=True)

        # # TEST
        # print('***** FOR TESTING FILE SAVED FORMAT *****')
        # model.save(os.path.join('/content/drive/MyDrive/fungi/ckpt', 'raw_model_for_testing.keras'))

        print('\n> Training Classification Head...')
        history = model.fit(train_batches,
                            epochs=args.pretrain_epoch,
                            validation_data=validation_batches,
                            callbacks=[early_stopper, reduce_lr])
        print(' >> Training Completed!')
        result = get_result(history)

        # save
        print('\n> Saving Model...')
        save_model_dir_path = os.path.join(args.ckpt_dir_path, 'model_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        os.mkdir(save_model_dir_path)
        print(' >> Saved Path: {}'.format(save_model_dir_path))
        model.save(os.path.join(save_model_dir_path,
                                'MobileNetV2_pretrained_ep{}_bs{}_lr{}.keras'.format(args.pretrain_epoch,
                                                                                     args.pretrain_batch_size,
                                                                                     args.pretrain_learning_rate)))
        with open(os.path.join(save_model_dir_path, 'result_pretrain.json'), 'w') as out:
            json.dump(result, out)

    # step 2
    if args.finetune:
        train_batches = train.shuffle(args.batch_shuffle_buffer_size).batch(args.finetune_batch_size).prefetch(tf.data.AUTOTUNE)

        if args.pretrain_model_path:
            # for continue training
            print('\n> Loading ckpt: {}'.format(args.pretrain_model_path))
            model = tf.keras.models.load_model(args.pretrain_model_path)
            model.trainable = True
            base_model = model.layers[0]
            base_model.trainable = True
            for layer in base_model.layers[:args.finetune_at]:
                layer.trainable =  False
        else:
            base_model.trainable = True
            for layer in base_model.layers[:args.finetune_at]:
                layer.trainable =  False

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      optimizer = tf.keras.optimizers.RMSprop(args.finetune_learning_rate),
                      metrics=['accuracy'])

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                        mode='max',
                                                        min_delta=args.min_delta,
                                                        patience=args.patience,
                                                        factor=args.factor,
                                                        verbose=1,
                                                        cooldown=args.cooldown,
                                                        min_lr=0.00000001)

        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                         mode='max',
                                                         min_delta=0.005,
                                                         patience=10,
                                                         verbose=1,
                                                         restore_best_weights=True)

        print('\n> Finetuning...')
        history_finetune = model.fit(train_batches,
                                 epochs=args.finetune_epoch,
                                 validation_data=validation_batches,
                                 callbacks=[early_stopper, reduce_lr])
        print(' >> Finetuning Completed!')

        result = get_result(history_finetune)

        # save
        print('\n> Saving Model...')
        if args.pretrain_model_path:
            save_model_dir_path = os.path.join(os.path.dirname(args.pretrain_model_path), 'tag_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            os.mkdir(save_model_dir_path)
            print(' >> Saved Path: {}'.format(save_model_dir_path))
            model.save(os.path.join(save_model_dir_path,
                                    'MobileNetV2_finetuned_ep{}_bs{}_lr{}.keras'.format(args.finetune_epoch,
                                                                                        args.finetune_batch_size,
                                                                                        args.finetune_learning_rate)))
            with open(os.path.join(save_model_dir_path, 'args.json'), 'w') as out:
                json.dump(vars(args), out)
            with open(os.path.join(save_model_dir_path, 'result_finetune.json'), 'w') as out:
                json.dump(result, out)
            label_map = builder.info.features["label"]._str2int
            with open(os.path.join(save_model_dir_path, 'label_map.json'), 'w') as out:
                json.dump(label_map, out)
        else:
            print(' >> Saved Path: {}'.format(save_model_dir_path))
            model.save(os.path.join(save_model_dir_path,
                                    'MobileNetV2_finetuned_ep{}_bs{}_lr{}.keras'.format(args.finetune_epoch,
                                                                                        args.finetune_batch_size,
                                                                                        args.finetune_learning_rate)))
            with open(os.path.join(save_model_dir_path, 'args.json'), 'w') as out:
                json.dump(vars(args), out)
            with open(os.path.join(save_model_dir_path, 'result_finetune.json'), 'w') as out:
                json.dump(result, out)
            label_map = builder.info.features["label"]._str2int
            with open(os.path.join(save_model_dir_path, 'label_map.json'), 'w') as out:
                json.dump(label_map, out)
        print(' >> Saving Completed!')

    if args.evaluate:
        print('\n> Evaluating Model...')
        if not args.model_path:
            args.model_path = save_model_dir_path
        model_weight_path = os.path.join(args.model_path,
                                         'MobileNetV2_finetuned_ep{}_bs{}_lr{}.keras'.format(args.finetune_epoch,
                                                                                             args.finetune_batch_size,
                                                                                             args.finetune_learning_rate))

        print(' >> Evaluating {}'.format(args.model_path))
        model = tf.keras.models.load_model(model_weight_path)
        top_n_result, gt_labels, predictions = get_top_n_acc(validation_batches, model)
        print(' \n>> top1:{}\n    top2:{}\n    top3:{}\n    top4:{}\n    top5:{}'.format(
            top_n_result['top_1_acc'],
            top_n_result['top_2_acc'],
            top_n_result['top_3_acc'],
            top_n_result['top_4_acc'],
            top_n_result['top_5_acc'],
        ))
        with open(os.path.join(args.model_path, 'top_5_acc.json'), mode='w') as out:
            json.dump(top_n_result, out)

        # # TODO: save gt_labels, predictions as a dictionary
        # eval_result = dict()
        # idx = 0
        # for gt_label, pred in zip(gt_labels, predictions):
        #     eval_result[idx] = {"gt": gt_label, "pred": pred}
        #     idx += 1
        # with open(os.path.join(args.model_path, 'eval_result.json'), mode='w') as out:
        #     json.dump(eval_result, out)

    if args.plot:
        plot(history, desc='step 1')
        plot(history_finetune, desc='step 2')

    elapsed_time = time.time() - start_time
    print('\n> Total Elaspsed Time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


if __name__== '__main__':
    main()