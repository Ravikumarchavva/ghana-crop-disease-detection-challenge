import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from configs import paths_helper

# Set up directoris for training a yolo model

# Images directories
from pathlib import Path
DATA_DIR = Path(paths_helper.DATA_DIR)
DATASET_DIR = DATA_DIR
IMAGES_DIR = DATASET_DIR / 'images'
TRAIN_IMAGES_DIR = IMAGES_DIR / 'train'
VAL_IMAGES_DIR = IMAGES_DIR / 'val'
TEST_IMAGES_DIR = IMAGES_DIR / 'test'

# Labels directories
LABELS_DIR = DATASET_DIR / 'labels'
TRAIN_LABELS_DIR = LABELS_DIR / 'train'
VAL_LABELS_DIR = LABELS_DIR / 'val'
TEST_LABELS_DIR = LABELS_DIR / 'test'

print(f'dataset_dir: {DATASET_DIR}')
print(f'images_dir: {IMAGES_DIR}')
print(f'label_dir: {LABELS_DIR}')

import pandas as pd
train = pd.read_csv(DATASET_DIR / 'Train.csv')

# Create a data.yaml file required by yolo
import yaml

class_names = sorted(train['class'].unique().tolist())
num_classes = len(class_names)

data_yaml = {
    'train': str(TRAIN_IMAGES_DIR),
    'val': str(VAL_IMAGES_DIR),
    'test': str(TEST_IMAGES_DIR),
    'nc': num_classes,
    'names': class_names
}

yaml_path = 'data.yaml'
with open(yaml_path, 'w') as file:
    yaml.dump(data_yaml, file, default_flow_style=False)

# Preview data yaml file
data_yaml