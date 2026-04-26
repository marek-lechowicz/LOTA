import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import cv2
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_NAME_MAP = {
    0: 'flux_1_dev',
    1: 'flux_fill_flux_1_dev',
    2: 'flux_fill_real_rescaled',
    3: 'flux_fill_sd_3_5_large',
    4: 'sd_1_5',
    5: 'sd_3_5_large',
    6: 'sdxl_turbo',
    7: 'z_image_turbo'
}

VAL_LIST_PATH = Path("/home/marek/FakeFlickr/data/flickr30k_entities/val.txt")
TEST_LIST_PATH = Path("/home/marek/FakeFlickr/data/flickr30k_entities/test.txt")

DATASET_PATHS = {
    "real": "/home/marek/FakeFlickr/data/fake-flickr/real",
    "real_rescaled": "/home/marek/FakeFlickr/data/fake-flickr/real_rescaled",
    "flux_1_dev": "/home/marek/FakeFlickr/data/fake-flickr/generated/flux_1_dev/img",
    "sd_3_5_large": "/home/marek/FakeFlickr/data/fake-flickr/generated/sd_3_5_large/img",
    "sdxl_turbo": "/home/marek/FakeFlickr/data/fake-flickr/generated/sdxl_turbo/img",
    "z_image_turbo": "/home/marek/FakeFlickr/data/fake-flickr/generated/z_image_turbo/img",
    "flux_fill_real_rescaled": "/home/marek/FakeFlickr/data/fake-flickr/generated/flux_fill_real_rescaled/img",
    "flux_fill_flux_1_dev": "/home/marek/FakeFlickr/data/fake-flickr/generated/flux_fill_flux_1_dev/img",
    "flux_fill_sd_3_5_large": "/home/marek/FakeFlickr/data/fake-flickr/generated/flux_fill_sd_3_5_large/img",
    "sd_1_5": "/home/marek/FakeFlickr/data/fake-flickr/generated/sd_1_5/img",
}

def read_split_ids(file_path: Path):
    if not file_path.exists():
        return set()
    return {
        line.strip().replace("\r", "")
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

def get_split_files(directory, split):
    """Deterministically split files into train, val, test sets matching flickr30k exact sets from data.py."""
    val_ids = read_split_ids(VAL_LIST_PATH)
    test_ids = read_split_ids(TEST_LIST_PATH)
    
    # Train set based exactly on base 'real' IDs missing from val and test
    base_real_dir = DATASET_PATHS["real"]
    base_files = [f for f in os.listdir(base_real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    base_real_ids = {os.path.splitext(f)[0] for f in base_files}
    train_ids = base_real_ids - val_ids - test_ids
    
    if split == 'val':
        target_ids = val_ids
    elif split == 'test':
        target_ids = test_ids
    elif split == 'train':
        target_ids = train_ids
    else:
        target_ids = base_real_ids
        
    # Read files in the targeted directory and keep only those matching target_ids
    if not os.path.exists(directory):
        print(f"Warning: directory {directory} does not exist.")
        return []

    files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    
    split_files = []
    for f in files:
        base_id = os.path.splitext(f)[0]
        if base_id in target_ids:
            split_files.append(f)
            
    return split_files

def create_augmentations(is_training=False):
    transforms_list = []
    
    if is_training:
        transforms_list.append(A.ImageCompression(quality_range=(70, 95), p=0.1))
        transforms_list.append(A.HorizontalFlip(p=0.5))
        transforms_list.append(A.RandomBrightnessContrast(p=0.2))
        transforms_list.append(A.GaussNoise(p=0.2))

    return A.Compose(transforms_list)


def create_normalization():
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def apply_preprocessing(image, options, is_training=False):
    image_np = np.array(image)
    
    # 1. Albumentations (szum, kompresja, flip) - tylko na PEŁNYM obrazie w f. treningu
    aug_pipeline = create_augmentations(is_training=is_training)
    augmented_full = aug_pipeline(image=image_np)
    image_np = augmented_full["image"]
    
    # 2. Extract Patch (szuka łatek na rozszerzonym i przetworzonym obrazie)
    if options.isPatch:
        patched_image = bit_patch_process(
            image_np, options.img_height, options.bit_mode,
            options.patch_size, options.patch_mode
        )
        image_np = np.array(patched_image)
    else:
        # Fallback jeśli patch_mode nie jest użyte (na wszelki wypadek)
        print("No patch mode used")
        image_np = cv2.resize(image_np, (options.img_height, options.img_height))
        
    # 3. Normalizacja i ToTensor() na ostatecznie wyciętym kawałku
    norm_pipeline = create_normalization()
    final_aug = norm_pipeline(image=image_np)
    return final_aug["image"]
class GenerativeImageTrainingSet(Dataset):
    def __init__(self, root_dir, dataset_name, options):
        super().__init__()
        self.options = options
        
        # In the new structure, we fetch absolute paths matching ResNet
        real_dir = DATASET_PATHS[options.real_source]
        ai_dir = DATASET_PATHS.get(dataset_name, os.path.join(root_dir, dataset_name))

        # Get deterministic split
        self.natural_filenames = get_split_files(real_dir, 'train')
        self.ai_filenames = get_split_files(ai_dir, 'train')

        self.natural_images = [os.path.join(real_dir, f) for f in self.natural_filenames]
        self.ai_images = [os.path.join(ai_dir, f) for f in self.ai_filenames]

        self.all_images = self.natural_images + self.ai_images
        self.labels = torch.cat([
            torch.ones(len(self.natural_images)),
            torch.zeros(len(self.ai_images))
        ])

    def _load_rgb(self, img_path):
        try:
            image = cv2.imread(img_path)
            if image is None:
                 return Image.new('RGB', (256, 256), (0, 0, 0))
            
            # Equalize format bias by compressing lossless images (PNG/WebP) to jpg with quality 90
            if not str(img_path).lower().endswith((".jpg", ".jpeg")):
                success, encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if success:
                    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        except Exception as e:
            print(f"Image Loading Error {img_path}: {str(e)}")
            return Image.new('RGB', (256, 256), (0, 0, 0))

    def __getitem__(self, index):
        try:
            img = self._load_rgb(self.all_images[index])
            label = self.labels[index]
        except:
            prev_index = max(0, index - 1)
            img = self._load_rgb(self.all_images[prev_index])
            label = self.labels[prev_index]

        processed_img = apply_preprocessing(img, self.options, is_training=True)
        return processed_img, label

    def __len__(self):
        return len(self.all_images)

class GenerativeImageValidationSet(Dataset):
    def __init__(self, root_dir, dataset_name, is_natural, options, split='val'):
        super().__init__()
        self.options = options
        
        if is_natural:
            self.img_dir = DATASET_PATHS[options.real_source]
        else:
            self.img_dir = DATASET_PATHS.get(dataset_name, os.path.join(root_dir, dataset_name))
        
        # Use 'val' or 'test' split depending on needs
        filenames = get_split_files(self.img_dir, split)
        self.image_paths = [os.path.join(self.img_dir, f) for f in filenames]

        self.labels = torch.ones(len(self.image_paths)) if is_natural else torch.zeros(len(self.image_paths))

    def _load_rgb(self, img_path):
        try:
            image = cv2.imread(img_path)
            if image is None:
                 return Image.new('RGB', (256, 256), (0, 0, 0))
                 
            # Equalize format bias by compressing lossless images (PNG/WebP) to jpg with quality 90
            if not str(img_path).lower().endswith((".jpg", ".jpeg")):
                success, encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if success:
                    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        except Exception as e:
            print(f"Val Image Loading Error {img_path}: {str(e)}")
            return Image.new('RGB', (256, 256), (0, 0, 0))

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = self._load_rgb(img_path)
        label = self.labels[index]

        processed_img = apply_preprocessing(img, self.options, is_training=False)
        return processed_img, label, img_path

    def __len__(self):
        return len(self.image_paths)

def create_validation_loader(options, dataset_name, is_natural):
    # Detect if we should use val or test split
    # Since test.py and train.py use same config flags, 
    # we can try to check if we are in main of test.py or something, 
    # but more robustly let's just use 'test' split if some flag is set.
    # For now, let's use 'val' as default.
    split = 'test' if (hasattr(options, 'isTest') and options.isTest) else 'val'
    
    val_dataset = GenerativeImageValidationSet(
        options.image_root, dataset_name, is_natural, options, split=split
    )

    def collate_batch(batch):
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return inputs, labels, paths

    return DataLoader(
        val_dataset,
        batch_size=options.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch
    ), len(val_dataset)


def setup_validation_loaders(options):
    choices = options.choices
    loaders = []

    for idx, selected in enumerate(choices):
        if selected:
            loader_info = {}
            model_name = MODEL_NAME_MAP[idx]
            print(f"Val dataset: {model_name}")

            loader_info['name'] = model_name
            loader_info['val_ai_loader'], loader_info['ai_size'] = create_validation_loader(
                options, model_name, False
            )
            loader_info['val_nature_loader'], loader_info['nature_size'] = create_validation_loader(
                options, model_name, True
            )

            loaders.append(loader_info)

    return loaders

def create_training_loader(options):
    choices = options.choices
    root_dir = options.image_root

    datasets = []

    dataset_config = [
        (0, "flux_1_dev"),
        (1, "flux_fill_flux_1_dev"),
        (2, "flux_fill_real_rescaled"),
        (3, "flux_fill_sd_3_5_large"),
        (4, "sd_1_5"),
        (5, "sd_3_5_large"),
        (6, "sdxl_turbo"),
        (7, "z_image_turbo")
    ]

    for idx, folder_name in dataset_config:
        if choices[idx]:
            dataset = GenerativeImageTrainingSet(
                root_dir, folder_name, options
            )
            datasets.append(dataset)
            print(f"Train dataset: {MODEL_NAME_MAP[idx]}")

    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    return DataLoader(
        combined_dataset,
        batch_size=options.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

def get_loader(opt):
    return create_training_loader(opt)

def get_val_loader(opt):
    return setup_validation_loaders(opt)

def get_single_loader(opt, image_dir, is_real):
    return create_validation_loader(opt, image_dir, is_real)

from bit_patch import bit_patch as bit_patch_process
