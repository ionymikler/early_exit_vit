# device_manager.py
import torch
import threading
from typing import Optional
from utils.logging_utils import get_logger_ready

logger = get_logger_ready(__name__)


class DeviceManager:
    _instance: Optional["DeviceManager"] = None
    _device: Optional[torch.device] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize only if not already initialized
        if DeviceManager._device is None:
            self._init_device()

    def _init_device(self):
        """Initialize the device with default settings (CUDA if available)"""
        DeviceManager._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"DeviceManager initialized with device: {DeviceManager._device}")

        if DeviceManager._device.type == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            )
            logger.info(
                f"CUDA Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
            )

    @classmethod
    def get_device(cls) -> torch.device:
        """Get the current device"""
        if cls._instance is None:
            cls._instance = DeviceManager()
        return cls._device

    @classmethod
    def set_device(cls, device: str) -> None:
        """
        Set the device to be used.
        Args:
            device: Either 'cuda', 'cpu', or a specific cuda device like 'cuda:0'
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = DeviceManager()

            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            cls._device = torch.device(device)
            logger.info(f"Device set to: {cls._device}")

    @classmethod
    def is_cuda_available(cls) -> bool:
        """Check if CUDA is available"""
        return torch.cuda.is_available()

    @classmethod
    def clear_cuda_cache(cls) -> None:
        """Clear CUDA cache if using CUDA device"""
        if cls._device is not None and cls._device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

    def __str__(self):
        return f"DeviceManager(device={self._device})"
