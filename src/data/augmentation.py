"""Data Augmentation for Handwriting Images.

Simulates real-world degradation:
- Camera blur (Gaussian, motion)
- Shadow and uneven lighting
- Glare/specular highlights
- JPEG compression artifacts
- Rotation and perspective
- Noise (salt & pepper, Gaussian)
- Color jitter
- Partial occlusion
"""

import random
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import structlog

logger = structlog.get_logger()


class HandwritingAugmentation:
    """Augmentation pipeline for handwriting images.
    
    Designed to simulate real-world phone-camera captures:
    - Students photographing their work
    - Varied lighting conditions
    - Different phone cameras and qualities
    - Paper creases and shadows
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        blur_prob: float = 0.3,
        shadow_prob: float = 0.2,
        rotation_prob: float = 0.3,
        noise_prob: float = 0.2,
        jpeg_prob: float = 0.2,
        color_jitter_prob: float = 0.3,
    ):
        self.probability = probability
        self.blur_prob = blur_prob
        self.shadow_prob = shadow_prob
        self.rotation_prob = rotation_prob
        self.noise_prob = noise_prob
        self.jpeg_prob = jpeg_prob
        self.color_jitter_prob = color_jitter_prob
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to an image."""
        if random.random() > self.probability:
            return image
        
        # Apply augmentations with individual probabilities
        if random.random() < self.blur_prob:
            image = self._apply_blur(image)
        
        if random.random() < self.rotation_prob:
            image = self._apply_rotation(image)
        
        if random.random() < self.noise_prob:
            image = self._apply_noise(image)
        
        if random.random() < self.jpeg_prob:
            image = self._apply_jpeg_compression(image)
        
        if random.random() < self.color_jitter_prob:
            image = self._apply_color_jitter(image)
        
        return image
    
    def _apply_blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur to simulate camera defocus."""
        radius = random.uniform(0.5, 2.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        """Apply small rotation to simulate tilted capture."""
        angle = random.uniform(-5, 5)
        return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    
    def _apply_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to simulate camera sensor noise."""
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def _apply_jpeg_compression(self, image: Image.Image) -> Image.Image:
        """Simulate JPEG compression artifacts."""
        import io
        quality = random.randint(30, 70)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    
    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        """Apply color/brightness/contrast jitter."""
        # Brightness
        factor = random.uniform(0.7, 1.3)
        image = ImageEnhance.Brightness(image).enhance(factor)
        
        # Contrast
        factor = random.uniform(0.7, 1.3)
        image = ImageEnhance.Contrast(image).enhance(factor)
        
        return image
