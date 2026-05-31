import json
import os
import random
from pathlib import Path
from typing import Union, List, Dict, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch 
import numpy as np

# Project base path (for saving models, outputs, etc.)
base_path = Path(__file__).absolute().parents[1].absolute()

def _resolve_dataset_path() -> Path:
    candidates = [
        os.environ.get("MULSETRANK_DATASETS_DIR"),
        "/workspace/CAM-CIR_backup/datasets",
        "/root/siton-data-92a7d2fc7b594215b07e48fd8818598b/CAM-CIR_backup/datasets",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return Path(candidates[-1]).expanduser()


# Dataset path (where the datasets are stored)
dataset_path = _resolve_dataset_path()
_FASHION200K_SPLIT_RECORDS_CACHE: Dict[str, List[dict]] = {}
_FASHION200K_TRAIN_STATE_CACHE: Optional[Tuple[Dict[str, List[int]], Dict[str, List[str]], List[int]]] = None
_SHOES_NAME_TO_PATH_CACHE: Optional[Dict[str, Path]] = None
_SHOES_TRIPLETS_CACHE: Dict[str, List[dict]] = {}

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(
        self,
        split: str,
        dress_types: List[str],
        mode: str,
        preprocess: callable,
        val_split_mode: Literal['val-split', 'original-split'] = 'original-split',
    ):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        :param val_split_mode: validation candidate pool mode, should be in ['val-split', 'original-split']:
            - 'val-split': only include images that appear in validation queries
            - 'original-split': include all images from the original validation image split
            Default is 'original-split'. Only takes effect when split == 'val' and mode == 'classic'.
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.val_split_mode = val_split_mode

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        if val_split_mode not in ['val-split', 'original-split']:
            raise ValueError("val_split_mode should be in ['val-split', 'original-split']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(dataset_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(dataset_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        if split == 'val' and mode == 'classic' and val_split_mode == 'val-split':
            query_image_names = set()
            for triplet in self.triplets:
                query_image_names.add(triplet['candidate'])
                query_image_names.add(triplet['target'])
            self.image_names = [image_name for image_name in self.image_names if image_name in query_image_names]
            print(f"Applied FashionIQ val-split filtering: {len(self.image_names)} images kept for retrieval")

        print(
            f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized "
            f"(val_split_mode: {val_split_mode})"
        )

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    reference_image_path = dataset_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = dataset_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_captions, target_name

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = dataset_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = dataset_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
    
    def get_image_by_name(self, image_name: str):
        """
        Load and preprocess an image by its name.
        Used for loading hard negative samples during training.
        
        Args:
            image_name: Name of the image (without extension)
            
        Returns:
            Preprocessed image tensor
        """
        image_path = dataset_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
        image = self.preprocess(PIL.Image.open(image_path))
        return image
    
    def get_images_by_names(self, image_names: List[str], use_parallel: bool = True, num_workers: int = 4):
        """
        Batch load and preprocess multiple images by their names (optimized).
        
        Args:
            image_names: List of image names (without extension)
            use_parallel: Whether to use parallel loading (faster for many images)
            num_workers: Number of parallel workers
            
        Returns:
            Stacked tensor of preprocessed images [N, C, H, W]
        """
        if not use_parallel or len(image_names) < 4:
            # Sequential loading for small batches
            images = [self.get_image_by_name(name) for name in image_names]
        else:
            # Parallel loading for large batches
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                images = list(executor.map(self.get_image_by_name, image_names))
        
        return torch.stack(images)
    
    def get_images_batch(self, image_names: List[str], num_workers: int = 4) -> torch.Tensor:
        """
        高效批量获取图像。
        
        Args:
            image_names: List of image names
            
        Returns:
            Stacked tensor of preprocessed images [N, C, H, W]
        """
        unique_names = list(dict.fromkeys(image_names))
        if len(unique_names) < 4:
            unique_images = [self.get_image_by_name(name) for name in unique_names]
        else:
            max_workers = min(max(1, num_workers), len(unique_names))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                unique_images = list(executor.map(self.get_image_by_name, unique_names))

        unique_image_map = dict(zip(unique_names, unique_images))
        images = [unique_image_map[name] for name in image_names]
        return torch.stack(images)


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(dataset_path / 'cirr_dataset' / 'CIRR' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(dataset_path / 'cirr_dataset' / 'CIRR' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = dataset_path / 'cirr_dataset' / 'CIRR' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = dataset_path / 'cirr_dataset' / 'CIRR' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, rel_caption, target_hard_name

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = dataset_path / 'cirr_dataset' / 'CIRR' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
    
    def get_image_by_name(self, image_name: str):
        """
        Load and preprocess an image by its name.
        Used for loading hard negative samples during training.
        
        Args:
            image_name: Name of the image
            
        Returns:
            Preprocessed image tensor
        """
        if image_name not in self.name_to_relpath:
            raise ValueError(f"Image name {image_name} not found in dataset")
        image_path = dataset_path / 'cirr_dataset' / 'CIRR' / self.name_to_relpath[image_name]
        image = self.preprocess(PIL.Image.open(image_path))
        return image
    
    def get_images_by_names(self, image_names: List[str], use_parallel: bool = True, num_workers: int = 4):
        """
        Batch load and preprocess multiple images by their names (optimized).
        
        Args:
            image_names: List of image names
            use_parallel: Whether to use parallel loading (faster for many images)
            num_workers: Number of parallel workers
            
        Returns:
            Stacked tensor of preprocessed images [N, C, H, W]
        """
        if not use_parallel or len(image_names) < 4:
            # Sequential loading for small batches
            images = [self.get_image_by_name(name) for name in image_names]
        else:
            # Parallel loading for large batches
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                images = list(executor.map(self.get_image_by_name, image_names))
        
        return torch.stack(images)
    
    def get_images_batch(self, image_names: List[str], num_workers: int = 4) -> torch.Tensor:
        """
        高效批量获取图像。
        
        Args:
            image_names: List of image names
            
        Returns:
            Stacked tensor of preprocessed images [N, C, H, W]
        """
        unique_names = list(dict.fromkeys(image_names))
        if len(unique_names) < 4:
            unique_images = [self.get_image_by_name(name) for name in unique_names]
        else:
            max_workers = min(max(1, num_workers), len(unique_names))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                unique_images = list(executor.map(self.get_image_by_name, unique_names))

        unique_image_map = dict(zip(unique_names, unique_images))
        images = [unique_image_map[name] for name in image_names]
        return torch.stack(images)


def _fashion200k_caption_post_process(caption: str) -> str:
    return (
        caption.strip()
        .replace(".", "dotmark")
        .replace("?", "questionmark")
        .replace("&", "andmark")
        .replace("*", "starmark")
    )


def _fashion200k_get_different_word(
    source_caption: str,
    target_caption: str,
) -> Optional[Tuple[str, str, str]]:
    source_words = source_caption.split()
    target_words = target_caption.split()

    source_word = next((word for word in source_words if word not in target_words), None)
    target_word = next((word for word in target_words if word not in source_words), None)
    if source_word is None or target_word is None:
        return None

    return source_word, target_word, f"replace {source_word} with {target_word}"


def _load_fashion200k_split_records(split: Literal["train", "test"]) -> List[dict]:
    if split in _FASHION200K_SPLIT_RECORDS_CACHE:
        return _FASHION200K_SPLIT_RECORDS_CACHE[split]

    split_records = []
    fashion200k_root = dataset_path / "fashion200k"

    for label_file in sorted((fashion200k_root / "labels").glob(f"*_{split}_detect_all.txt")):
        with open(label_file) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue

                file_path, detection_score, caption = parts
                full_path = fashion200k_root / file_path
                split_records.append(
                    {
                        "file_path": file_path,
                        "full_path": full_path,
                        "detection_score": float(detection_score),
                        "caption": _fashion200k_caption_post_process(caption),
                        "exists": full_path.exists(),
                    }
                )

    _FASHION200K_SPLIT_RECORDS_CACHE[split] = split_records
    return split_records


def _build_fashion200k_sampling_state(records: List[dict]) -> Tuple[Dict[str, List[int]], Dict[str, List[str]], List[int]]:
    caption2imgids: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        record["parent_captions"] = []
        record["valid_parent_captions"] = []
        if not record["exists"]:
            continue
        caption2imgids.setdefault(record["caption"], []).append(idx)

    parent2children_captions: Dict[str, List[str]] = {}
    for caption in caption2imgids.keys():
        for word in caption.split():
            parent_caption = " ".join(caption.replace(word, "").split())
            parent2children_captions.setdefault(parent_caption, [])
            if caption not in parent2children_captions[parent_caption]:
                parent2children_captions[parent_caption].append(caption)

    for parent_caption, candidate_captions in parent2children_captions.items():
        if len(candidate_captions) < 2:
            continue
        for caption in candidate_captions:
            for img_idx in caption2imgids[caption]:
                records[img_idx]["parent_captions"].append(parent_caption)

    valid_source_indices = []
    for idx, record in enumerate(records):
        if not record["exists"]:
            continue

        for parent_caption in record["parent_captions"]:
            candidate_captions = parent2children_captions[parent_caption]
            has_valid_target = False
            for target_caption in candidate_captions:
                if target_caption == record["caption"]:
                    continue
                if not caption2imgids.get(target_caption):
                    continue
                if _fashion200k_get_different_word(record["caption"], target_caption) is not None:
                    has_valid_target = True
                    break

            if has_valid_target:
                record["valid_parent_captions"].append(parent_caption)

        if record["valid_parent_captions"]:
            valid_source_indices.append(idx)

    return caption2imgids, parent2children_captions, valid_source_indices


def _get_fashion200k_train_sampling_state() -> Tuple[Dict[str, List[int]], Dict[str, List[str]], List[int]]:
    global _FASHION200K_TRAIN_STATE_CACHE
    if _FASHION200K_TRAIN_STATE_CACHE is None:
        train_records = _load_fashion200k_split_records("train")
        _FASHION200K_TRAIN_STATE_CACHE = _build_fashion200k_sampling_state(train_records)
    return _FASHION200K_TRAIN_STATE_CACHE


class Fashion200kDataset(Dataset):
    """
    Fashion200k dataset for standard CIR training/evaluation.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yields (image_name, image)
        - In 'relative' mode the dataset yields:
            - (reference_image, target_image, rel_caption, target_name) when split == train
            - (reference_name, target_name, rel_caption) when split == val
    """

    def __init__(self, split: str, mode: str, preprocess: callable, seed: int = 0):
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.seed = seed
        self.data_root = dataset_path / "fashion200k"

        if split not in ["train", "val"]:
            raise ValueError("split should be in ['train', 'val']")
        if mode not in ["relative", "classic"]:
            raise ValueError("mode should be in ['relative', 'classic']")

        split_name = "train" if split == "train" else "test"
        self.records = _load_fashion200k_split_records(split_name)
        self.name_to_path = {
            record["file_path"]: record["full_path"]
            for record in self.records
            if record["exists"]
        }

        if self.mode == "classic":
            self.image_names = [record["file_path"] for record in self.records if record["exists"]]
            print(
                f"Fashion200k {split} dataset in classic mode initialized "
                f"with {len(self.image_names)} images"
            )
            return

        if split == "train":
            (
                self.caption2imgids,
                self.parent2children_captions,
                self.valid_source_indices,
            ) = _get_fashion200k_train_sampling_state()
            if not self.valid_source_indices:
                raise RuntimeError("No valid Fashion200k training sources found")
            print(
                f"Fashion200k train dataset in relative mode initialized "
                f"with {len(self.records)} source slots and {len(self.valid_source_indices)} valid sources"
            )
        else:
            caption_by_name = {
                record["file_path"]: record["caption"]
                for record in self.records
                if record["exists"]
            }
            self.triplets = []
            skipped_queries = 0
            with open(self.data_root / "test_queries.txt") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    reference_name, target_name = parts
                    if reference_name not in caption_by_name or target_name not in caption_by_name:
                        skipped_queries += 1
                        continue

                    diff = _fashion200k_get_different_word(
                        caption_by_name[reference_name],
                        caption_by_name[target_name],
                    )
                    if diff is None:
                        skipped_queries += 1
                        continue

                    self.triplets.append(
                        {
                            "reference": reference_name,
                            "target": target_name,
                            "caption": diff[2],
                            "index": idx,
                        }
                    )

            print(
                f"Fashion200k val dataset in relative mode initialized "
                f"with {len(self.triplets)} queries (skipped {skipped_queries})"
            )

    def _resolve_train_source_index(self, index: int) -> int:
        if index < len(self.records):
            record = self.records[index]
            if record["exists"] and record.get("valid_parent_captions"):
                return index
        return self.valid_source_indices[index % len(self.valid_source_indices)]

    def _sample_train_triplet(self, index: int) -> Tuple[str, str, str]:
        source_index = self._resolve_train_source_index(index)
        source_record = self.records[source_index]
        rng = random.Random(self.seed + index)
        source_caption = source_record["caption"]

        while True:
            parent_caption = rng.choice(source_record["valid_parent_captions"])
            target_caption = rng.choice(self.parent2children_captions[parent_caption])
            if target_caption == source_caption:
                continue

            diff = _fashion200k_get_different_word(source_caption, target_caption)
            if diff is None:
                continue

            target_candidates = self.caption2imgids.get(target_caption, [])
            if not target_candidates:
                continue

            target_index = rng.choice(target_candidates)
            target_record = self.records[target_index]
            return source_record["file_path"], target_record["file_path"], diff[2]

    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                if self.split == "train":
                    reference_name, target_name, rel_caption = self._sample_train_triplet(index)
                    reference_image = self.preprocess(PIL.Image.open(self.name_to_path[reference_name]))
                    target_image = self.preprocess(PIL.Image.open(self.name_to_path[target_name]))
                    return reference_image, target_image, rel_caption, target_name

                triplet = self.triplets[index]
                return triplet["reference"], triplet["target"], triplet["caption"]

            image_name = self.image_names[index]
            image = self.preprocess(PIL.Image.open(self.name_to_path[image_name]))
            return image_name, image

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            if self.split == "train":
                return len(self.records)
            return len(self.triplets)
        return len(self.image_names)

    def get_image_by_name(self, image_name: str):
        if image_name not in self.name_to_path:
            raise ValueError(f"Image name {image_name} not found in Fashion200k dataset")
        return self.preprocess(PIL.Image.open(self.name_to_path[image_name]))

    def get_images_by_names(self, image_names: List[str], use_parallel: bool = True, num_workers: int = 4):
        if not use_parallel or len(image_names) < 4:
            images = [self.get_image_by_name(name) for name in image_names]
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                images = list(executor.map(self.get_image_by_name, image_names))
        return torch.stack(images)

    def get_images_batch(self, image_names: List[str], num_workers: int = 4) -> torch.Tensor:
        unique_names = list(dict.fromkeys(image_names))
        if len(unique_names) < 4:
            unique_images = [self.get_image_by_name(name) for name in unique_names]
        else:
            max_workers = min(max(1, num_workers), len(unique_names))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                unique_images = list(executor.map(self.get_image_by_name, unique_names))

        unique_image_map = dict(zip(unique_names, unique_images))
        images = [unique_image_map[name] for name in image_names]
        return torch.stack(images)


def _build_shoes_name_to_path() -> Dict[str, Path]:
    global _SHOES_NAME_TO_PATH_CACHE
    if _SHOES_NAME_TO_PATH_CACHE is not None:
        return _SHOES_NAME_TO_PATH_CACHE

    shoes_root = dataset_path / "shoes_dataset" / "shoes_data"
    name_to_path = {}
    for image_path in shoes_root.rglob("*.jpg"):
        name_to_path[image_path.name] = image_path
    _SHOES_NAME_TO_PATH_CACHE = name_to_path
    return name_to_path


def _load_shoes_unique_triplets(
    split: Literal["train", "test"],
    name_to_path: Dict[str, Path],
) -> List[dict]:
    if split in _SHOES_TRIPLETS_CACHE:
        return _SHOES_TRIPLETS_CACHE[split]

    shoes_root = dataset_path / "shoes_dataset" / "shoes_data"
    pairs_path = shoes_root / ("relative_pairs_train.npy" if split == "train" else "relative_pairs_test.npy")
    pairs = np.load(pairs_path, allow_pickle=True)

    triplets = []
    seen_triplets = set()
    for item in pairs:
        reference_name = Path(item["source"]).name
        target_name = Path(item["target"]).name
        rel_caption = item["mod"]
        triplet_key = (reference_name, target_name, rel_caption)
        if triplet_key in seen_triplets:
            continue
        if reference_name not in name_to_path or target_name not in name_to_path:
            continue
        seen_triplets.add(triplet_key)
        triplets.append(
            {
                "reference": reference_name,
                "target": target_name,
                "caption": rel_caption,
            }
        )

    _SHOES_TRIPLETS_CACHE[split] = triplets
    return triplets


class ShoesDataset(Dataset):
    """
    Shoes dataset for standard CIR training/evaluation.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yields (image_name, image)
        - In 'relative' mode the dataset yields:
            - (reference_image, target_image, rel_caption, target_name) when split == train
            - (reference_name, target_name, rel_caption) when split == val
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.data_root = dataset_path / "shoes_dataset" / "shoes_data"

        if split not in ["train", "val"]:
            raise ValueError("split should be in ['train', 'val']")
        if mode not in ["relative", "classic"]:
            raise ValueError("mode should be in ['relative', 'classic']")

        self.name_to_path = _build_shoes_name_to_path()

        if self.mode == "classic":
            split_file = self.data_root / ("train_im_names.txt" if split == "train" else "eval_im_names.txt")
            with open(split_file) as f:
                self.image_names = [line.strip() for line in f if line.strip() and line.strip() in self.name_to_path]
            print(
                f"Shoes {split} dataset in classic mode initialized "
                f"with {len(self.image_names)} images"
            )
            return

        split_name = "train" if split == "train" else "test"
        self.triplets = _load_shoes_unique_triplets(split_name, self.name_to_path)
        print(
            f"Shoes {split} dataset in relative mode initialized "
            f"with {len(self.triplets)} triplets"
        )

    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                triplet = self.triplets[index]
                reference_name = triplet["reference"]
                target_name = triplet["target"]
                rel_caption = triplet["caption"]

                if self.split == "train":
                    reference_image = self.preprocess(PIL.Image.open(self.name_to_path[reference_name]))
                    target_image = self.preprocess(PIL.Image.open(self.name_to_path[target_name]))
                    return reference_image, target_image, rel_caption, target_name

                return reference_name, target_name, rel_caption

            image_name = self.image_names[index]
            image = self.preprocess(PIL.Image.open(self.name_to_path[image_name]))
            return image_name, image

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        return len(self.image_names)

    def get_image_by_name(self, image_name: str):
        if image_name not in self.name_to_path:
            raise ValueError(f"Image name {image_name} not found in Shoes dataset")
        return self.preprocess(PIL.Image.open(self.name_to_path[image_name]))

    def get_images_by_names(self, image_names: List[str], use_parallel: bool = True, num_workers: int = 4):
        if not use_parallel or len(image_names) < 4:
            images = [self.get_image_by_name(name) for name in image_names]
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                images = list(executor.map(self.get_image_by_name, image_names))
        return torch.stack(images)

    def get_images_batch(self, image_names: List[str], num_workers: int = 4) -> torch.Tensor:
        unique_names = list(dict.fromkeys(image_names))
        if len(unique_names) < 4:
            unique_images = [self.get_image_by_name(name) for name in unique_names]
        else:
            max_workers = min(max(1, num_workers), len(unique_names))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                unique_images = list(executor.map(self.get_image_by_name, unique_names))

        unique_image_map = dict(zip(unique_names, unique_images))
        images = [unique_image_map[name] for name in image_names]
        return torch.stack(images)


class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(PIL.Image.open(img_path))
            return {
                'img': img,
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
