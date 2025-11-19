"""
特徵提取共用工具模組
統一管理 MediaPipe Pool、視頻預載入、設備檢測等
"""

from .mediapipe_pool import (
    MediaPipePosePool,
    MediaPipeHolisticPool,
    SelfieSegmentationPool,
    _is_jupyter_environment
)
from .video_preloader import VideoPreloader, AsyncBatchPreloader
from .common import (
    get_device,
    create_zip,
    setup_logging,
    detect_apple_silicon,
    setup_apple_silicon_optimization
)

__all__ = [
    'MediaPipePosePool',
    'MediaPipeHolisticPool',
    'SelfieSegmentationPool',
    'VideoPreloader',
    'AsyncBatchPreloader',
    'get_device',
    'create_zip',
    'setup_logging',
    'detect_apple_silicon',
    'setup_apple_silicon_optimization',
    '_is_jupyter_environment'
]
