#!/usr/bin/env python3
"""
骨架流特徵提取 - MediaPipe Holistic（上半身 + 手指細節）
專為 M1 MPS Metal 或 CPU 優化的手語視頻骨架特徵提取
- 上半身姿態（11 個關鍵點）
- 雙手細節（21 點 × 2 = 42 個關鍵點）
- 總計 53 個關鍵點，159 維特徵
- 多進程 + 多線程 CPU 優化
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import zipfile
from datetime import datetime
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import psutil

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("❌ 請安裝 mediapipe: pip install mediapipe") from e


class EnhancedSkeletonExtractor:
    """增強版骨架提取器 - MediaPipe Holistic（上半身 + 手指細節）"""
    
    # MediaPipe Pose 上半身關鍵點索引
    UPPER_BODY_INDICES = [
        0,   # nose
        2,   # left_eye_inner
        5,   # right_eye_inner
        11,  # left_shoulder
        12,  # right_shoulder
        13,  # left_elbow
        14,  # right_elbow
        15,  # left_wrist
        16,  # right_wrist
        23,  # left_hip (軀幹參考)
        24   # right_hip (軀幹參考)
    ]
    
    def __init__(self, num_threads=None):
        """
        初始化提取器 - MediaPipe 自動使用 M1 Metal 加速

        Args:
            num_threads: 線程數，None 表示自動檢測
        """
        # 檢測系統和可用資源
        self._detect_system_resources(num_threads)

        # 初始化 MediaPipe Holistic (強制使用 CPU)
        print("📥 載入 MediaPipe Holistic 模型...")
        print("   模式: CPU (強制)")
        print(f"   線程數: {self.num_threads}")

        # 強制使用 CPU (禁用所有 GPU/OpenGL 加速)
        # 這些環境變數必須在導入 mediapipe 之前設置，但為了容器環境額外確保
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA GPU
        os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'  # 禁用 MediaPipe GPU
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # 替代環境變數
        
        # 禁用 OpenGL/EGL (關鍵！Zeabur 容器無 GPU 支援)
        os.environ['GLOG_minloglevel'] = '2'  # 減少錯誤日誌
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 減少 TensorFlow 日誌
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # 保持 Full 版本以維持特徵品質
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,  # 保持高品質檢測
            min_tracking_confidence=0.5
        )

        print("✅ 模型載入完成")

    def _detect_system_resources(self, num_threads):
        """檢測系統資源和優化設置"""
        try:
            import torch
            self.mps_available = torch.backends.mps.is_available()
        except:
            self.mps_available = False

        # 檢測 CPU 核心數
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cpu_count = psutil.cpu_count(logical=False)

        # 設置線程數 (強制CPU模式)
        if num_threads is None:
            # CPU模式：使用更多線程但留核心給系統
            self.num_threads = max(1, self.physical_cpu_count - 1)
        else:
            self.num_threads = num_threads

        print(f"🔍 系統資源檢測:")
        print(f"   - CPU 核心: {self.cpu_count} (物理: {self.physical_cpu_count})")
        print(f"   - 模式: CPU (強制)")
        print(f"   - 優化線程數: {self.num_threads}")

        # 設置 OpenMP 線程數（影響 MediaPipe）
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)

        # 關鍵點名稱（用於調試）
        self.upper_body_names = [
            'nose', 'left_eye_inner', 'right_eye_inner',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip'
        ]

        print(f"📊 特徵配置:")
        print(f"   - 上半身關鍵點: {len(self.UPPER_BODY_INDICES)} 個")
        print(f"   - 左手關鍵點: 21 個")
        print(f"   - 右手關鍵點: 21 個")
        print(f"   - 總關鍵點: {len(self.UPPER_BODY_INDICES) + 42} 個")
        print(f"   - 總特徵維度: {(len(self.UPPER_BODY_INDICES) + 42) * 3} 維")
    
    def normalize_skeleton_features(self, upper_body, left_hand, right_hand, frame_width, frame_height):
        """
        正規化骨架特徵
        
        Args:
            upper_body: (T, 11, 3) - 上半身關鍵點
            left_hand: (T, 21, 3) - 左手關鍵點
            right_hand: (T, 21, 3) - 右手關鍵點
            frame_width: 幀寬度
            frame_height: 幀高度
        
        Returns:
            normalized: (T, 159) - 正規化後的特徵
        """
        T = len(upper_body)
        
        # 組合所有關鍵點 (T, 53, 3)
        all_keypoints = np.concatenate([upper_body, left_hand, right_hand], axis=1)
        all_keypoints = all_keypoints.astype(np.float32)
        
        normalized = all_keypoints.copy()
        
        # Step 1: 座標正規化 [0, 1]
        # MediaPipe 已經輸出正規化座標，但我們再確保一次
        normalized[:, :, 0] = np.clip(normalized[:, :, 0], 0, 1)  # x
        normalized[:, :, 1] = np.clip(normalized[:, :, 1], 0, 1)  # y
        normalized[:, :, 2] = np.clip(normalized[:, :, 2], -1, 1)  # z (深度)
        
        # Step 2: 置中處理 - 以肩膀中心為原點
        for t in range(T):
            # 上半身索引：left_shoulder=3, right_shoulder=4 (在組合後的數組中)
            left_shoulder = normalized[t, 3]
            right_shoulder = normalized[t, 4]
            
            # 檢查置信度（MediaPipe 的 visibility）
            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                center_y = (left_shoulder[1] + right_shoulder[1]) / 2
                
                # 所有點相對於肩膀中心
                normalized[t, :, 0] -= center_x
                normalized[t, :, 1] -= center_y
        
        # Step 3: 尺度不變性 - 根據肩寬正規化
        for t in range(T):
            left_shoulder = normalized[t, 3]
            right_shoulder = normalized[t, 4]
            
            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                shoulder_width = np.linalg.norm(
                    left_shoulder[:2] - right_shoulder[:2]
                )
                
                if shoulder_width > 0.01:
                    # 所有 x,y 座標除以肩寬
                    normalized[t, :, :2] /= shoulder_width
        
        # Step 4: 處理缺失關鍵點
        # MediaPipe 的 visibility < 0.3 視為不可見
        low_confidence_mask = normalized[:, :, 2] < 0.3
        normalized[low_confidence_mask, :2] = 0
        
        # Step 5: Flatten - (T, 159)
        return normalized.reshape(T, -1)
    
    def extract_features_from_frames(self, frames, frame_width=640, frame_height=480):
        """
        直接從 frames 提取骨架特徵（優化版，無需讀取視頻文件）
        
        Args:
            frames: list of numpy arrays (H, W, 3) - RGB frames
            frame_width: 幀寬度（用於正規化）
            frame_height: 幀高度（用於正規化）
        
        Returns:
            features: (T, 159) numpy array
        """
        try:
            if frames is None or len(frames) == 0:
                print(f"⚠️  沒有提供 frames")
                return None
            
            # 存儲關鍵點
            upper_body_keypoints = []
            left_hand_keypoints = []
            right_hand_keypoints = []
            
            # 逐幀處理
            for frame in frames:
                # 確保是 RGB 格式
                if frame.shape[2] == 3:
                    # 如果是 BGR，轉換為 RGB
                    # 假設已經是 RGB（從錄影直接來）
                    frame_rgb = frame
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # MediaPipe 處理
                results = self.holistic.process(frame_rgb)
                
                # 提取上半身關鍵點
                if results.pose_landmarks:
                    pose = results.pose_landmarks.landmark
                    upper_body = np.array([
                        [pose[i].x, pose[i].y, pose[i].visibility]
                        for i in self.UPPER_BODY_INDICES
                    ], dtype=np.float32)
                else:
                    upper_body = np.zeros((11, 3), dtype=np.float32)
                
                upper_body_keypoints.append(upper_body)
                
                # 提取左手關鍵點
                if results.left_hand_landmarks:
                    left_hand = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark
                    ], dtype=np.float32)
                else:
                    left_hand = np.zeros((21, 3), dtype=np.float32)
                
                left_hand_keypoints.append(left_hand)
                
                # 提取右手關鍵點
                if results.right_hand_landmarks:
                    right_hand = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark
                    ], dtype=np.float32)
                else:
                    right_hand = np.zeros((21, 3), dtype=np.float32)
                
                right_hand_keypoints.append(right_hand)
            
            if len(upper_body_keypoints) == 0:
                print(f"⚠️  未提取到任何幀")
                return None
            
            # 轉換為 numpy array
            upper_body_keypoints = np.array(upper_body_keypoints, dtype=np.float32)
            left_hand_keypoints = np.array(left_hand_keypoints, dtype=np.float32)
            right_hand_keypoints = np.array(right_hand_keypoints, dtype=np.float32)
            
            # 正規化
            features = self.normalize_skeleton_features(
                upper_body_keypoints,
                left_hand_keypoints,
                right_hand_keypoints,
                frame_width,
                frame_height
            )
            
            return features  # (T, 159)
        
        except Exception as e:
            import traceback
            print(f"❌ 從 frames 提取特徵失敗: {e}")
            print(f"詳細錯誤: {traceback.format_exc()}")
            return None
    
    def extract_features(self, video_path):
        """提取單個視頻的增強骨架特徵"""
        try:
            # 開啟視頻
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"⚠️  無法開啟視頻: {video_path.name}")
                return None
            
            # 獲取視頻資訊
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if total_frames == 0:
                cap.release()
                print(f"⚠️  視頻無幀: {video_path.name}")
                return None
            
            # 讀取所有幀
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 轉換為 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            # 使用新方法提取特徵
            return self.extract_features_from_frames(frames, width, height)
        
        except Exception as e:
            import traceback
            print(f"❌ 處理失敗 {video_path.name}: {e}")
            print(f"詳細錯誤: {traceback.format_exc()}")
            return None
    
    def process_directory(self, input_dir, output_dir):
        """處理整個目錄 - 多進程CPU優化"""
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
            "feature_dim": 159,
            "keypoint_breakdown": {
                "upper_body": 11,
                "left_hand": 21,
                "right_hand": 21,
                "total": 53
            },
            "cpu_optimization": {
                "num_threads": self.num_threads,
                "mode": "CPU (強制)",
                "cpu_cores": self.cpu_count
            }
        }

        print(f"📝 發現 {len(word_dirs)} 個單詞目錄")
        print(f"🚀 使用多進程優化: {self.num_threads} 線程")

        # 處理每個單詞目錄
        for word_dir in tqdm(word_dirs, desc="處理單詞"):
            word_name = word_dir.name
            word_output_dir = output_path / word_name
            word_output_dir.mkdir(exist_ok=True, parents=True)

            # 獲取所有視頻文件
            video_files = list(word_dir.glob("*.mp4"))
            word_stats = {"word": word_name, "total": len(video_files), "success": 0, "failed": 0}

            # 使用線程池處理單詞內的視頻文件
            def process_video_batch(video_batch):
                """處理一批視頻文件"""
                local_success = 0
                local_failed = 0

                for video_file in video_batch:
                    try:
                        # 提取特徵
                        features = self.extract_features(video_file)

                        if features is not None:
                            # 保存 .npy
                            output_file = word_output_dir / f"{video_file.stem}.npy"
                            np.save(output_file, features)
                            local_success += 1
                        else:
                            local_failed += 1
                    except Exception as e:
                        print(f"⚠️  處理失敗 {video_file.name}: {e}")
                        local_failed += 1

                return local_success, local_failed

            # 將視頻文件分成批次進行處理
            batch_size = max(1, len(video_files) // self.num_threads)
            video_batches = [video_files[i:i + batch_size] for i in range(0, len(video_files), batch_size)]

            # 使用線程池處理批次
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(process_video_batch, batch) for batch in video_batches]

                # 收集結果
                for future in futures:
                    success, failed = future.result()
                    word_stats["success"] += success
                    word_stats["failed"] += failed

            stats["total_videos"] += len(video_files)
            stats["successful"] += word_stats["success"]
            stats["failed"] += word_stats["failed"]
            stats["word_details"].append(word_stats)

        return stats
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


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
    print("🚀 骨架流特徵提取 - MediaPipe Holistic（上半身 + 手指細節）")
    print("=" * 60)

    # 路徑配置
    kaggle_input = Path("/kaggle/input/original-seg/videos_segmented")
    kaggle_working = Path("/kaggle/working")

    if kaggle_input.exists():
        print("🏠 檢測到 Kaggle 環境")
        input_dir = kaggle_input
        output_dir = kaggle_working / "skeleton_features"
        num_threads = None  # 自動檢測
    else:
        print("💻 本地環境模式")
        input_dir = Path("/Users/baidongqu/Desktop/MVP/videos")
        output_dir = Path("/Users/baidongqu/Desktop/MVP/skeleton_features")
        num_threads = None  # 自動檢測CPU核心數

    if not input_dir.exists():
        print(f"❌ 輸入目錄不存在: {input_dir}")
        return

    # 初始化提取器 - MediaPipe 自動優化
    start_time = datetime.now()
    extractor = EnhancedSkeletonExtractor(num_threads=num_threads)
    
    # 處理目錄
    stats = extractor.process_directory(input_dir, output_dir)
    
    # 打包 ZIP
    if stats["successful"] > 0:
        zip_path = create_zip(output_dir, "skeleton_features.zip")
        stats["zip_path"] = str(zip_path)
    
    # 保存報告
    stats["processing_time"] = str(datetime.now() - start_time)
    report_path = output_dir / "extraction_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 輸出統計
    print("\n" + "=" * 60)
    print("🎉 骨架特徵提取完成！")
    print(f"📊 總視頻數: {stats['total_videos']}")
    print(f"✅ 成功: {stats['successful']}")
    print(f"❌ 失敗: {stats['failed']}")
    print(f"⏱️  總耗時: {stats['processing_time']}")
    print(f"📐 特徵維度: {stats['feature_dim']}")
    print(f"   - 上半身: {stats['keypoint_breakdown']['upper_body']} 點")
    print(f"   - 左手: {stats['keypoint_breakdown']['left_hand']} 點")
    print(f"   - 右手: {stats['keypoint_breakdown']['right_hand']} 點")
    print(f"📋 報告: {report_path}")
    if "zip_path" in stats:
        print(f"📦 ZIP: {stats['zip_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
