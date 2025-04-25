"""3D UNet inference pipeline for lung nodule segmentation."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray


class NoduleSegmentor:
    """Wrapper for loading a 3D UNet model and performing inference."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize segmentation model on the given device.

        Args:
            model_path: Path to PyTorch .pt model file
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()
        logging.info(f"Loaded model on {self.device}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        try:
            model = torch.load(model_path, map_location=self.device, weights_only=False)  # noqa: E501
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError("Invalid model format")
            return model
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")  # noqa: B904

    def predict(
        self, volume: NDArray[np.float32], threshold: float = 0.5, batch_size: int = 4
    ) -> NDArray[np.uint8]:
        """Segment nodules from 3D CT volume.

        Args:
            volume: 3D volume [D, H, W] (preprocessed)
            threshold: Probability threshold for binarizing output
            batch_size: For memory-constrained chunking (if depth > 128)

        Returns:
            Binary mask [D, H, W] as np.uint8
        """
        input_tensor = (
            torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        if input_tensor.size(2) > 128:
            return self._predict_by_chunks(input_tensor, threshold, batch_size)

        with torch.no_grad():
            output = self.model(input_tensor)
            return (
                (torch.sigmoid(output) > threshold)
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

    def _predict_by_chunks(
        self, tensor: torch.Tensor, threshold: float, batch_size: int
    ) -> NDArray[np.uint8]:
        """Placeholder for large volume chunked inference."""
        raise NotImplementedError("Chunked prediction not implemented yet.")
