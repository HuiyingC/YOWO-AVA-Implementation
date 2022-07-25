# Created by Huiying Chen, 21 July 2022
# Make sample datasets for training

import os
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('max_columns', None)
pd.set_option('max_colwidth', None)


def extract_frame_lists(video_IDs, mode):
    frame_list_train_dir = 'ava_action/frames/train.csv'
    frame_list_val_dir   = 'ava_action/frames/val.csv'

    if mode == 'train':
        df_train = pd.read_csv(frame_list_train_dir, sep=' ')
        print(df_train.columns)
        df_train_sample = df_train.loc[df_train['original_vido_id'].isin(video_IDs)]
        # df_train_sample = df_train_sample.fillna(' ')
        print(df_train_sample.head())
        df_train_sample.to_csv('ava_action/frame_lists/train_sample.csv', index=False)

    if mode == 'val':
        df_val = pd.read_csv(frame_list_val_dir, sep=' ')
        print(df_val.columns)
        df_val_sample = df_val.loc[df_val['original_vido_id'].isin(video_IDs)]
        print(df_val_sample)
        df_val_sample.to_csv('ava_action/frame_lists/val_sample.csv', index=False)


def extract_boxes_and_labels(video_IDs, mode):
    boxes_and_labels_train_dir = 'ava_action/ava_annotations_full/person_box_67091280_iou90/' \
                                 'ava_detection_train_boxes_and_labels_include_negative_v2.2.csv'
    boxes_and_labels_val_dir   = 'ava_action/ava_annotations_full/person_box_67091280_iou90/' \
                                 'ava_detection_val_boxes_and_labels.csv'

    if mode == 'train':
        df_train = pd.read_csv(boxes_and_labels_train_dir)
        df_train.columns = ['vido_id', 'frame_id', 'label_1', 'label_2', 'label_3', 'label_4',
                            'class', 'score']
        print(df_train.columns)
        df_train_sample = df_train.loc[df_train['vido_id'].isin(video_IDs)]
        print(df_train_sample)
        df_train_sample.to_csv('ava_action/annotations_sample/'
                               'ava_detection_train_boxes_and_labels_include_negative_v2.2.csv',
                               index=False)

    if mode == 'val':
        df_val = pd.read_csv(boxes_and_labels_val_dir)
        df_val.columns = ['vido_id', 'frame_id', 'label_1', 'label_2', 'label_3', 'label_4',
                            'class', 'score']
        print(df_val.columns)
        df_val_sample = df_val.loc[df_val['vido_id'].isin(video_IDs)]
        print(df_val_sample)
        df_val_sample.to_csv('ava_action/annotations_sample/'
                             'ava_detection_val_boxes_and_labels.csv',
                             index=False)


def extract_train_val_v2(video_IDs, mode):
    train_dir = 'ava_action/ava_annotations_full/ava_train_v2.2.csv'
    val_dir   = 'ava_action/ava_annotations_full/ava_val_v2.2.csv'

    if mode == 'train':
        df_train = pd.read_csv(train_dir)
        df_train.columns = ['vido_id', 'frame_id', 'label_1', 'label_2', 'label_3', 'label_4',
                            'class_1', 'class_2']
        print(df_train.columns)
        df_train_sample = df_train.loc[df_train['vido_id'].isin(video_IDs)]
        print(df_train_sample)
        df_train_sample.to_csv('ava_action/annotations_sample'
                               '/ava_train_v2.2.csv',
                               index=False)

    if mode == 'val':
        df_val = pd.read_csv(val_dir)
        df_val.columns = ['vido_id', 'frame_id', 'label_1', 'label_2', 'label_3', 'label_4',
                            'class_1', 'class_2']
        print(df_val.columns)
        df_val_sample = df_val.loc[df_val['vido_id'].isin(video_IDs)]
        print(df_val_sample)
        df_val_sample.to_csv('ava_action/annotations_sample'
                               '/ava_val_v2.2.csv',
                               index=False)


def extract_predicted_boxes(video_IDs, mode):
    predicted_boxes_val_dir = 'ava_action/ava_annotations_full/person_box_67091280_iou90/ava_val_predicted_boxes.csv'

    if mode == 'val':
        df_val = pd.read_csv(predicted_boxes_val_dir)
        df_val.columns = ['vido_id', 'frame_id', 'label_1', 'label_2', 'label_3', 'label_4',
                            'class', 'score']
        print(df_val.columns)
        df_val_sample = df_val.loc[df_val['vido_id'].isin(video_IDs)]
        print(df_val_sample)
        df_val_sample.to_csv('ava_action/annotations_sample/'
                             'ava_val_predicted_boxes.csv',
                             index=False)




if __name__ == '__main__':
    train_video_IDs = ['N5UD8FGzDek']
    val_video_IDs = ['1j20qq1JyX4']

    extract_frame_lists(train_video_IDs, 'train')
    # extract_frame_lists(val_video_IDs, 'val')

    # extract_boxes_and_labels(train_video_IDs, 'train')
    # extract_boxes_and_labels(val_video_IDs, 'val')

    # extract_train_val_v2(train_video_IDs, 'train')
    # extract_train_val_v2(val_video_IDs, 'val')

    # extract_predicted_boxes(val_video_IDs, 'val')









