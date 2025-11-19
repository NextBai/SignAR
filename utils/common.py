#!/usr/bin/env python3
"""
通用工具函數
包含設備檢測、檔案打包、日誌設定等
"""

import os
import zipfile
import logging
from pathlib import Path


def get_device():
    """
    自動檢測設備：TPU > MPS > GPU > CPU

    Returns:
        (device, device_type) - PyTorch 設備對象和類型字串
    """
    try:
        import torch
    except ImportError:
        raise ImportError("請安裝 PyTorch: pip install torch")

    # 檢測 TPU
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        device_type = 'tpu'
        print(f"🚀 使用 TPU v5e-8")
        return device, device_type
    except ImportError:
        pass

    # 檢測 Apple Silicon MPS
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'
        print(f"🚀 使用 Apple Silicon GPU (Metal Performance Shaders)")
        return device, device_type

    # 檢測 CUDA GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 使用 GPU: {gpu_name}")
        return device, device_type

    # 降級到 CPU
    device = torch.device('cpu')
    device_type = 'cpu'
    print("⚠️  使用 CPU（速度較慢）")
    return device, device_type


def create_zip(output_dir, zip_name, file_pattern='*.npy'):
    """
    打包輸出目錄為 ZIP

    Args:
        output_dir: 輸出目錄路徑
        zip_name: ZIP 檔案名稱
        file_pattern: 要打包的檔案模式（預設 *.npy）

    Returns:
        zip_path: ZIP 檔案路徑
    """
    output_path = Path(output_dir)
    zip_path = output_path.parent / zip_name

    print(f"📦 打包輸出檔案到 {zip_path}...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_path.rglob(file_pattern):
            arcname = file_path.relative_to(output_path.parent)
            zipf.write(file_path, arcname)

    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ 打包完成！ZIP 檔案大小: {zip_size:.2f} MB")
    return zip_path


def setup_logging(log_file=None, level=logging.INFO):
    """
    設定日誌系統

    Args:
        log_file: 日誌檔案路徑（可選）
        level: 日誌等級（預設 INFO）
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )


def detect_apple_silicon():
    """
    檢測是否運行在 Apple Silicon (M1/M2/M3) 上

    Returns:
        bool: 是否為 Apple Silicon
    """
    import platform
    import subprocess

    try:
        # 檢查系統類型
        if platform.system() != 'Darwin':
            return False

        # 使用 sysctl 檢查處理器型號
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True,
            text=True,
            check=False
        )

        cpu_brand = result.stdout.lower()
        is_apple_silicon = 'apple' in cpu_brand or 'm1' in cpu_brand or 'm2' in cpu_brand or 'm3' in cpu_brand

        if is_apple_silicon:
            print(f"✅ 檢測到 Apple Silicon 晶片: {cpu_brand.strip()}")

        return is_apple_silicon

    except Exception:
        return False


def setup_apple_silicon_optimization():
    """
    設定 Apple Silicon 優化環境變數
    """
    if not detect_apple_silicon():
        return

    import psutil

    physical_cpu_count = psutil.cpu_count(logical=False)

    # 啟用 MediaPipe Metal 加速
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'
    os.environ['OMP_DYNAMIC'] = 'TRUE'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 優化內存分配
    os.environ['OPENBLAS_NUM_THREADS'] = str(physical_cpu_count)
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    print(f"✅ Apple Silicon 優化已啟用")
