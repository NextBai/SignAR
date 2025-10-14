#!/usr/bin/env python3
"""
RGB流特徵提取 - MobileNetV3
TPU 優先，GPU 降級，使用 MobileNetV3 提取視覺特徵
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import zipfile
from datetime import datetime
import json

try:
    import torchvision.models as models
    import torchvision.transforms as transforms
except ImportError as e:
    raise ImportError("❌ 請安裝 torchvision: pip install torchvision") from e


def get_device():
    """自動檢測設備：TPU > GPU > MPS > CPU"""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        device_type = 'tpu'
        print(f"🚀 使用 TPU v5e-8")
        return device, device_type
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_type = 'gpu'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 使用 GPU: {gpu_name}")
            return device, device_type
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            device_type = 'gpu'
            print(f"🚀 使用 Apple Silicon GPU (MPS)")
            return device, device_type
        else:
            device = torch.device('cpu')
            device_type = 'cpu'
            print("⚠️  使用 CPU（速度較慢）")
            return device, device_type


class RGBFeatureExtractor:
    """RGB 特徵提取器 - 基於 MobileNetV3"""
    
    def __init__(self, device, device_type):
        self.device = device
        self.device_type = device_type
        
        # 載入 MobileNetV3-Large 預訓練模型
        print("📥 載入 MobileNetV3-Large 模型...")
        try:
            # 新版 torchvision (>= 0.13)
            from torchvision.models import MobileNet_V3_Large_Weights
            self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # 舊版 torchvision
            self.model = models.mobilenet_v3_large(pretrained=True)
        
        # 移除分類層，保留特徵提取部分
        self.model.classifier = nn.Identity()
        
        self.model = self.model.to(device)
        self.model.eval()
        print("✅ 模型載入完成")
        
        # ImageNet 標準化參數
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.batch_size = 32 if device_type == 'tpu' else 16
    
    def load_video(self, video_path, max_frames=300):
        """讀取視頻幀，最多300幀"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frames = []
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # 轉換為 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
        
        cap.release()
        return frames
    
    def extract_features_batch(self, frames):
        """批次提取特徵"""
        all_features = []
        
        # 批次處理
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            
            # 預處理
            batch_tensors = torch.stack([
                self.transform(frame) for frame in batch_frames
            ]).to(self.device)
            
            # 前向傳播
            with torch.no_grad():
                features = self.model(batch_tensors)  # (batch, 960)
            
            all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)  # (T, 960)
    
    def normalize_rgb_features(self, features):
        """
        正規化 RGB 特徵
        features: (T, 960) - T幀的MobileNetV3特徵
        """
        # Step 1: L2正規化 - 每幀特徵向量單位化
        l2_norm = np.linalg.norm(features, axis=1, keepdims=True)
        l2_norm = np.where(l2_norm == 0, 1, l2_norm)  # 避免除以零
        features_l2 = features / l2_norm
        
        # Step 2: 時序標準化 - 減去視頻級別的均值和標準差
        mean = features_l2.mean(axis=0, keepdims=True)
        std = features_l2.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # 避免除以零
        features_normalized = (features_l2 - mean) / std
        
        # Step 3: Clip極端值 - 限制在[-3, 3]標準差範圍內
        features_normalized = np.clip(features_normalized, -3, 3)
        
        return features_normalized
    
    def extract_features_from_frames(self, frames):
        """
        直接從 frames 提取 RGB 特徵（優化版，無需讀取視頻文件）
        
        Args:
            frames: list of numpy arrays (H, W, 3) - RGB frames
        
        Returns:
            features: (T, 960) numpy array
        """
        try:
            if frames is None or len(frames) == 0:
                return None
            
            # 批次提取特徵
            features = self.extract_features_batch(frames)
            
            # 正規化
            features = self.normalize_rgb_features(features)
            
            return features
        
        except Exception as e:
            print(f"❌ 從 frames 提取特徵失敗: {e}")
            return None
    
    def extract_features(self, video_path):
        """提取單個視頻的 RGB 特徵"""
        try:
            # 讀取視頻
            frames = self.load_video(video_path)
            
            if frames is None or len(frames) == 0:
                return None
            
            # 使用新方法提取特徵
            return self.extract_features_from_frames(frames)
        
        except Exception as e:
            print(f"❌ 處理失敗 {video_path.name}: {e}")
            return None
    
    def process_directory(self, input_dir, output_dir):
        """處理整個目錄"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 獲取所有單詞目錄
        word_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        
        stats = {
            "total_videos": 0,
            "successful": 0,
            "failed": 0,
            "word_details": [],
            "device": self.device_type
        }
        
        print(f"📝 發現 {len(word_dirs)} 個單詞目錄")
        
        # 處理每個單詞目錄
        for word_dir in tqdm(word_dirs, desc="處理單詞"):
            word_name = word_dir.name
            word_output_dir = output_path / word_name
            word_output_dir.mkdir(exist_ok=True, parents=True)
            
            # 獲取所有視頻文件
            video_files = list(word_dir.glob("*.mp4"))
            word_stats = {"word": word_name, "total": len(video_files), "success": 0, "failed": 0}
            
            for video_file in tqdm(video_files, desc=f"  {word_name}", leave=False):
                stats["total_videos"] += 1
                
                # 提取特徵
                features = self.extract_features(video_file)
                
                if features is not None:
                    # 保存 .npy（檔名不變）
                    output_file = word_output_dir / f"{video_file.stem}.npy"
                    np.save(output_file, features)
                    stats["successful"] += 1
                    word_stats["success"] += 1
                else:
                    stats["failed"] += 1
                    word_stats["failed"] += 1
            
            stats["word_details"].append(word_stats)
        
        return stats


def create_zip(output_dir, zip_name):
    """打包輸出目錄為 ZIP"""
    output_path = Path(output_dir)
    zip_path = output_path.parent / zip_name
    
    print(f"📦 打包輸出檔案到 {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_path.rglob('*.npy'):
            arcname = file_path.relative_to(output_path.parent)
            zipf.write(file_path, arcname)
    
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ 打包完成！ZIP 檔案大小: {zip_size:.2f} MB")
    return zip_path


def main():
    """主函數"""
    print("🚀 RGB流特徵提取 - MobileNetV3")
    print("=" * 60)
    
    # 路徑配置
    kaggle_input = Path("/kaggle/input/original-seg/videos_segmented")
    kaggle_working = Path("/kaggle/working")
    
    if kaggle_input.exists():
        print("🏠 檢測到 Kaggle 環境")
        input_dir = kaggle_input
        output_dir = kaggle_working / "rgb_features"
    else:
        print("💻 本地環境模式")
        input_dir = Path("/Users/baidongqu/Desktop/SignAR/videos_segmented")
        output_dir = Path("/Users/baidongqu/Desktop/SignAR/rgb_features")
    
    if not input_dir.exists():
        print(f"❌ 輸入目錄不存在: {input_dir}")
        return
    
    # 檢測設備
    device, device_type = get_device()
    
    # 初始化提取器
    start_time = datetime.now()
    extractor = RGBFeatureExtractor(device, device_type)
    
    # 處理目錄
    stats = extractor.process_directory(input_dir, output_dir)
    
    # 打包 ZIP
    if stats["successful"] > 0:
        zip_path = create_zip(output_dir, "rgb_features.zip")
        stats["zip_path"] = str(zip_path)
    
    # 保存報告
    stats["processing_time"] = str(datetime.now() - start_time)
    report_path = output_dir / "extraction_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 輸出統計
    print("\n" + "=" * 60)
    print("🎉 RGB特徵提取完成！")
    print(f"📊 總視頻數: {stats['total_videos']}")
    print(f"✅ 成功: {stats['successful']}")
    print(f"❌ 失敗: {stats['failed']}")
    print(f"⏱️  總耗時: {stats['processing_time']}")
    print(f"📋 報告: {report_path}")
    if "zip_path" in stats:
        print(f"📦 ZIP: {stats['zip_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

