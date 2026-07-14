"""Dataset loaders for handwriting recognition.

Supports:
- IAM Handwriting Database
- SMHD (Student Messy Handwriting Dataset)
- GNHK (GoodNotes Handwriting Kollection)
- Custom datasets in standard format
"""

import csv
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import structlog

logger = structlog.get_logger()


class HandwritingDataset(Dataset):
    """Generic handwriting dataset for TrOCR training.
    
    Expected directory structure:
    data_dir/
        images/
            0001.png
            0002.png
            ...
        labels.csv  (columns: filename, text, writer_id)
    """
    
    def __init__(
        self,
        data_dir: str,
        processor=None,
        max_length: int = 128,
        augmentation: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.augmentation = augmentation
        
        # Load labels
        self.samples = self._load_labels()
        
        logger.info(
            "dataset.loaded",
            data_dir=str(data_dir),
            num_samples=len(self.samples),
        )
    
    def _load_labels(self) -> list[dict]:
        """Load image paths and labels from CSV."""
        labels_file = self.data_dir / "labels.csv"
        samples = []
        
        if not labels_file.exists():
            logger.warning("dataset.no_labels", path=str(labels_file))
            return samples
        
        with open(labels_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = self.data_dir / "images" / row["filename"]
                if image_path.exists():
                    samples.append({
                        "image_path": str(image_path),
                        "text": row["text"],
                        "writer_id": row.get("writer_id", "unknown"),
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        
        # Apply augmentation
        if self.augmentation is not None:
            image = self.augmentation(image)
        
        # Process image
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze()
        
        # Process text labels
        labels = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        
        # Replace padding token id with -100 for loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class IAMDataset(HandwritingDataset):
    """IAM Handwriting Database loader.
    
    The IAM database has a specific structure:
    - 657 writers
    - 1,539 pages
    - 13,353 text lines
    - Writer-independent splits provided
    """
    
    def _load_labels(self) -> list[dict]:
        """Load IAM-specific format (words.txt or lines.txt)."""
        samples = []
        
        # Try lines.txt format
        lines_file = self.data_dir / "lines.txt"
        if lines_file.exists():
            with open(lines_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split(" ")
                    if len(parts) >= 9:
                        line_id = parts[0]
                        status = parts[1]
                        if status == "ok":
                            text = " ".join(parts[8:]).replace("|", " ")
                            # Construct image path
                            parts_id = line_id.split("-")
                            img_path = (
                                self.data_dir / "lines" /
                                parts_id[0] /
                                f"{parts_id[0]}-{parts_id[1]}" /
                                f"{line_id}.png"
                            )
                            if img_path.exists():
                                writer_id = parts_id[0]
                                samples.append({
                                    "image_path": str(img_path),
                                    "text": text,
                                    "writer_id": writer_id,
                                })
        
        return samples


class GNHKDataset(HandwritingDataset):
    """GNHK (GoodNotes Handwriting Kollection) loader.
    
    Camera-captured handwritten text with:
    - 687 images
    - 9,363 lines
    - 172,936 characters
    - JSON annotations with word-level bounding boxes
    """
    
    def _load_labels(self) -> list[dict]:
        """Load GNHK JSON annotations."""
        import json
        
        samples = []
        annotations_dir = self.data_dir / "annotations"
        images_dir = self.data_dir / "images"
        
        if not annotations_dir.exists():
            return super()._load_labels()
        
        for json_file in sorted(annotations_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            
            image_file = images_dir / f"{json_file.stem}.jpg"
            if not image_file.exists():
                image_file = images_dir / f"{json_file.stem}.png"
            
            if not image_file.exists():
                continue
            
            # Extract word-level crops (done during preprocessing)
            for word_data in data.get("words", []):
                word_image_path = (
                    self.data_dir / "crops" / 
                    f"{json_file.stem}_{word_data['id']}.png"
                )
                if word_image_path.exists():
                    samples.append({
                        "image_path": str(word_image_path),
                        "text": word_data["text"],
                        "writer_id": data.get("writer_id", "unknown"),
                    })
        
        return samples
