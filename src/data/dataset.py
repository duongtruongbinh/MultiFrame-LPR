"""MultiFrameDataset for license plate recognition with multi-frame input."""
import os
import glob
import json
import random
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import get_train_transforms, get_val_transforms, get_degradation_transforms


class MultiFrameDataset(Dataset):
    """Dataset for multi-frame license plate recognition.
    
    Handles both real LR images and synthetic LR (degraded HR) images.
    Implements Scenario-B specific validation splitting logic.
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory containing track folders.
            mode: 'train' or 'val'.
            split_ratio: Train/val split ratio.
            img_height: Target image height.
            img_width: Target image width.
            char2idx: Character to index mapping.
            val_split_file: Path to validation split JSON file.
            seed: Random seed for reproducible splitting.
        """
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        
        if mode == 'train':
            self.transform = get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
        
        selected_tracks = train_tracks if mode == 'train' else val_tracks
        print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
        
        self._index_samples(selected_tracks)
        print(f"-> Total: {len(self.samples)} samples.")

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one with Scenario-B priority."""
        train_tracks, val_tracks = [], []
        
        # 1. Load split file if exists
        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            
            # Check consistency: If val empty or no Scenario-B, recreate
            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                print("⚠️ Split file invalid or missing Scenario-B. Recreating...")
                val_tracks = []  # Reset to trigger new split logic

        # 2. Create new split if needed
        if not val_tracks:
            print("⚠️ Creating new split (Taking Val only from Scenario-B)...")
            
            # Filter Scenario-B tracks
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
            
            if not scenario_b_tracks:
                print("⚠️ Warning: No 'Scenario-B' folder found. Using random from all.")
                scenario_b_tracks = all_tracks
            
            # Val size = (1 - split_ratio) * total_scenario_b
            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
            
            # Shuffle and take from beginning as val
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]
            
            # Train = (All) - (Val)
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]
            
            # Save track IDs (folder names)
            os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
            with open(self.val_split_file, 'w') as f:
                json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        """Index all samples from selected tracks."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                
                track_id = os.path.basename(track_path)
                
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                # Sample 1: Real LR (guaranteed 5 frames)
                self.samples.append({
                    'paths': lr_files,
                    'label': label,
                    'is_synthetic': False,
                    'track_id': track_id
                })
                
                # Sample 2: Synthetic LR (only in train mode, guaranteed 5 frames)
                if self.mode == 'train':
                    self.samples.append({
                        'paths': hr_files,
                        'label': label,
                        'is_synthetic': True,
                        'track_id': track_id
                    })
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        """Load exactly 5 frames (guaranteed by dataset structure)."""
        item = self.samples[idx]
        img_paths = item['paths']
        label = item['label']
        is_synthetic = item['is_synthetic']
        track_id = item['track_id']
        
        images_list = []
        for p in img_paths:
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if is_synthetic and self.degrade:
                image = self.degrade(image=image)['image']
            
            image = self.transform(image=image)['image']
            images_list.append(image)

        images_tensor = torch.stack(images_list, dim=0)
        target = [self.char2idx[c] for c in label if c in self.char2idx]
        if len(target) == 0:
            target = [0]
            
        return images_tensor, torch.tensor(target, dtype=torch.long), len(target), label, track_id

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...], Tuple[str, ...]]:
        """Custom collate function for DataLoader."""
        images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text, track_ids
