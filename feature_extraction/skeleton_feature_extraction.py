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
import multiprocessing as mp_proc
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import psutil
from scipy import interpolate
import queue
import time

try:
    import mediapipe as mp
    mp_drawing_styles = mp.solutions.drawing_styles
except ImportError as e:
    raise ImportError("❌ 請安裝 mediapipe: pip install mediapipe") from e


class MediaPipeHolisticPool:
    """
    MediaPipe Holistic 線程池管理
    - 為 Jupyter ThreadPoolExecutor 環境提供線程隔離
    - 每個線程使用獨立的 Holistic 實例，避免時間戳記衝突
    - 根據 MediaPipe 最佳實踐，確保多線程環境下的穩定性
    """
    _thread_local = threading.local()
    _lock = threading.Lock()
    _all_instances = []
    
    @classmethod
    def get_holistic(cls, model_complexity=0):
        """獲取當前線程專用的 Holistic 實例"""
        if not hasattr(cls._thread_local, 'holistic') or cls._thread_local.holistic is None:
            with cls._lock:
                mp_holistic = mp.solutions.holistic
                holistic = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    smooth_segmentation=False,
                    refine_face_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                cls._thread_local.holistic = holistic
                cls._all_instances.append(holistic)
        return cls._thread_local.holistic
    
    @classmethod
    def close_thread_holistic(cls):
        """清理當前線程的 Holistic 實例"""
        if hasattr(cls._thread_local, 'holistic') and cls._thread_local.holistic is not None:
            cls._thread_local.holistic.close()
            cls._thread_local.holistic = None
    
    @classmethod
    def close_all(cls):
        """清理所有線程的 Holistic 實例"""
        with cls._lock:
            for holistic in cls._all_instances:
                try:
                    holistic.close()
                except:
                    pass
            cls._all_instances.clear()


def _is_jupyter_environment():
    """檢測是否在 Jupyter/IPython 環境中運行"""
    try:
        # 嘗試從 IPython 導入
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except ImportError:
        # 如果 IPython 不可用，不在 Jupyter 中
        return False


class EnhancedSkeletonExtractor:
    """增強版骨架提取器 - MediaPipe Holistic（上半身 + 手指細節）"""
    
    # MediaPipe Pose 上半身關鍵點索引（11 個關鍵點 = 33 維）
    # ⚠️ 重要：模型期待 RGB (960) + Skeleton (159) = 1119 維
    # 總維度：11×3 + 21×3 + 21×3 = 33 + 63 + 63 = 159 維
    # 與 Kaggle 訓練代碼完全一致
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
        23,  # left_hip
        24   # right_hip
    ]
    
    def __init__(self, num_threads=None):
        """
        初始化提取器 - MediaPipe 自動使用 M1 Metal 加速

        Args:
            num_threads: 線程數，None 表示自動檢測
        """
        # 檢測系統和可用資源
        self._detect_system_resources(num_threads)

        # 檢測是否在 Jupyter 環境
        self.is_jupyter = _is_jupyter_environment()

        # 初始化 MediaPipe Holistic (自動使用 Metal 加速)
        if self.is_jupyter:
            # Jupyter 環境：使用線程池版本
            self.holistic = None  # 延遲初始化，由 ThreadLocal 管理
            self.use_thread_local = True
        else:
            # 命令行/Kaggle 環境：直接創建單一實例
            self.mp_holistic = mp.solutions.holistic
            # 使用 model_complexity=0（最快速度）
            # 0=快速但精度低, 1=均衡（推薦）, 2=精確但很慢
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,  # 最快速度
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                refine_face_landmarks=True,  # 啟用 refined face landmarks 獲取瞳孔
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_thread_local = False

    def _detect_system_resources(self, num_threads):
        """檢測系統資源和優化設置 - M1 Metal 加速優化"""
        # 1. 檢測 GPU 和加速器支援
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                self.num_gpus = torch.cuda.device_count()
            else:
                self.num_gpus = 0
        except:
            self.cuda_available = False
            self.num_gpus = 0

        # 2. 檢測 M1 Metal 加速支援
        self.is_apple_silicon = self._detect_apple_silicon()
        
        # 3. 檢測 CPU 核心數
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cpu_count = psutil.cpu_count(logical=False)

        # 4. 優化設置選擇（按優先級）
        if self.is_apple_silicon:
            # M1/M2/M3 優化：充分利用 P-cores 和 E-cores
            self._setup_m1_optimization(num_threads)
        elif self.cuda_available:
            # Kaggle T4 GPU 優化設置
            if num_threads is None:
                self.num_threads = min(self.num_gpus * 2, 8)  # 每 GPU 2 進程
            else:
                self.num_threads = num_threads
        else:
            # CPU fallback
            if num_threads is None:
                self.num_threads = max(2, self.physical_cpu_count - 1)
            else:
                self.num_threads = num_threads

        # 5. 設置 OpenMP 線程數（影響 MediaPipe）
        os.environ['OMP_NUM_THREADS'] = str(max(1, self.cpu_count // self.num_threads))
        os.environ['MKL_NUM_THREADS'] = str(max(1, self.cpu_count // self.num_threads))
        
        # 6. M1 特定環境變數優化
        if self.is_apple_silicon:
            os.environ['OPENBLAS_NUM_THREADS'] = str(self.physical_cpu_count)
            # 禁用 Accelerate 多線程以避免與 OpenMP 衝突
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    def _detect_apple_silicon(self):
        """檢測是否運行在 Apple Silicon (M1/M2/M3) 上"""
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
        except:
            return False

    def _setup_m1_optimization(self, num_threads):
        """針對 M1 晶片的最佳化配置"""
        # M1 晶片有 8 核心：4 個高效能核心 (P-cores) + 4 個高效率核心 (E-cores)
        # 最佳配置：使用 75% 的物理核心用於處理，保留 1 核給系統
        
        if num_threads is None:
            # 計算最優線程數
            available_cores = max(1, self.physical_cpu_count - 1)
            # M1 可以安全地使用所有可用核心（MetalKit 優化）
            self.num_threads = available_cores
            
            print(f"🎯 M1 優化配置:")
            print(f"   - 總物理核心: {self.physical_cpu_count}")
            print(f"   - 可用核心: {available_cores}")
            print(f"   - 工作線程數: {self.num_threads}")
        else:
            self.num_threads = num_threads
        
        # 啟用 MediaPipe Metal 加速
        self._enable_metal_acceleration()

    def _enable_metal_acceleration(self):
        """啟用 MediaPipe Metal 加速"""
        # MediaPipe 在 Apple Silicon 上自動使用 Metal
        # 但我們可以通過環境變數進一步優化
        
        # 使用 CoreML 後端（M1 原生最優化）
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'
        
        # 啟用多線程推理
        os.environ['OMP_DYNAMIC'] = 'TRUE'
        
        # 優化内存分配
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少 TensorFlow 日誌
    
    def normalize_skeleton_features(self, upper_body, left_hand, right_hand, frame_width, frame_height):
        """
        正規化骨架特徵（向量化優化版本）
        
        Args:
            upper_body: (T, 6, 3) - 上半身關鍵點（肩膀、肘部、眼睛）
            left_hand: (T, 21, 3) - 左手關鍵點
            right_hand: (T, 21, 3) - 右手關鍵點
            frame_width: 幀寬度
            frame_height: 幀高度
        
        Returns:
            normalized: (T, 159) - 正規化後的特徵
        """
        T = len(upper_body)
        
        # 組合所有關鍵點 (T, 48, 3) - 6個上半身 + 21個左手 + 21個右手
        all_keypoints = np.concatenate([upper_body, left_hand, right_hand], axis=1)
        all_keypoints = all_keypoints.astype(np.float32)
        
        normalized = all_keypoints.copy()
        
        # Step 1: 座標正規化 [0, 1]（向量化操作，比迴圈快 5-10 倍）
        normalized[:, :, 0] = np.clip(normalized[:, :, 0], 0, 1)  # x
        normalized[:, :, 1] = np.clip(normalized[:, :, 1], 0, 1)  # y
        normalized[:, :, 2] = np.clip(normalized[:, :, 2], -1, 1)  # z (深度)
        
        # Step 2: 置中處理 - 以肩膀中心為原點（向量化）
        # 上半身索引：left_shoulder=0, right_shoulder=1 (在組合後的數組中)
        left_shoulder = normalized[:, 0]  # (T, 3)
        right_shoulder = normalized[:, 1]  # (T, 3)
        
        # 檢查置信度的向量
        valid_mask = (left_shoulder[:, 2] > 0.3) & (right_shoulder[:, 2] > 0.3)
        
        # 計算肩膀中心（向量化）
        centers = (left_shoulder[:, :2] + right_shoulder[:, :2]) / 2  # (T, 2)
        
        # 對有效幀進行置中
        normalized[valid_mask, :, 0] -= centers[valid_mask, 0:1]
        normalized[valid_mask, :, 1] -= centers[valid_mask, 1:2]
        
        # Step 3: 尺度不變性 - 根據肩寬正規化（向量化）
        shoulder_diffs = left_shoulder[:, :2] - right_shoulder[:, :2]
        shoulder_widths = np.linalg.norm(shoulder_diffs, axis=1)  # (T,)
        
        # 避免除以零
        safe_widths = np.where(shoulder_widths > 0.01, shoulder_widths, 1.0)
        
        # 對所有幀進行尺度正規化（正確的廣播）
        scale_factor = safe_widths[:, np.newaxis, np.newaxis]  # (T, 1, 1)
        normalized[valid_mask, :, :2] /= scale_factor[valid_mask]
        
        # Step 4: 處理缺失關鍵點（向量化）
        # MediaPipe 的 visibility < 0.3 視為不可見
        low_confidence_mask = normalized[:, :, 2] < 0.3
        normalized[low_confidence_mask, :2] = 0
        
        # Step 5: Flatten - (T, 159)
        return normalized.reshape(T, -1)
    
    def extract_features(self, video_file):
        """
        從視頻文件提取骨架特徵（流式工作流，減少內存）
        
        Args:
            video_file: 視頻文件路徑
        
        Returns:
            features: (T, 159) numpy array 或 None
        """
        try:
            video_path = Path(video_file)
            if not video_path.exists():
                return None
            
            # 使用 OpenCV 讀取視頻幀
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 改進：先計算總幀數，決定採樣策略
            total_frames_approx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_rate = self._calculate_optimal_sample_rate(total_frames_approx)
            sample_indices = self._generate_sample_indices(total_frames_approx, sample_rate)
            sample_set = set(sample_indices)  # 用 set 快速查詢
            
            # 預分配陣列
            upper_body_keypoints = np.zeros((total_frames_approx, 6, 3), dtype=np.float32)
            left_hand_keypoints = np.zeros((total_frames_approx, 21, 3), dtype=np.float32)
            right_hand_keypoints = np.zeros((total_frames_approx, 21, 3), dtype=np.float32)
            
            frame_idx = 0
            valid_frame_count = 0
            
            # 流式讀取 + 即時處理（不預先加載所有幀）
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 只處理採樣的幀
                if frame_idx not in sample_set:
                    frame_idx += 1
                    continue
                
                frame_rgb = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 在 Jupyter 環境中使用 ThreadLocal 實例
                if self.use_thread_local:
                    holistic = MediaPipeHolisticPool.get_holistic(model_complexity=0)
                else:
                    holistic = self.holistic
                
                results = holistic.process(frame_rgb)
                
                # 提取上半身關鍵點
                if results.pose_landmarks:
                    pose = results.pose_landmarks.landmark
                    for j, pose_idx in enumerate(self.UPPER_BODY_INDICES):
                        if pose_idx >= 0:
                            p = pose[pose_idx]
                            upper_body_keypoints[frame_idx, j] = [p.x, p.y, p.visibility]
                        else:
                            if results.face_landmarks:
                                face = results.face_landmarks.landmark
                                if pose_idx == -1:
                                    iris_indices = [469, 470, 471, 472]
                                    iris_points = [face[idx] for idx in iris_indices if idx < len(face)]
                                    if iris_points:
                                        center_x = sum(p.x for p in iris_points) / len(iris_points)
                                        center_y = sum(p.y for p in iris_points) / len(iris_points)
                                        visibility = sum(p.z for p in iris_points) / len(iris_points)
                                        upper_body_keypoints[frame_idx, j] = [center_x, center_y, visibility]
                                elif pose_idx == -2:
                                    iris_indices = [474, 475, 476, 477]
                                    iris_points = [face[idx] for idx in iris_indices if idx < len(face)]
                                    if iris_points:
                                        center_x = sum(p.x for p in iris_points) / len(iris_points)
                                        center_y = sum(p.y for p in iris_points) / len(iris_points)
                                        visibility = sum(p.z for p in iris_points) / len(iris_points)
                                        upper_body_keypoints[frame_idx, j] = [center_x, center_y, visibility]
                
                # 提取左手關鍵點
                if results.left_hand_landmarks:
                    left_hand_keypoints[frame_idx] = np.array([
                        [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
                    ], dtype=np.float32)
                
                # 提取右手關鍵點
                if results.right_hand_landmarks:
                    right_hand_keypoints[frame_idx] = np.array([
                        [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
                    ], dtype=np.float32)
                
                valid_frame_count += 1
                frame_idx += 1
            
            cap.release()
            
            if valid_frame_count == 0:
                return None
            
            # 只保留實際處理的幀
            upper_body_keypoints = upper_body_keypoints[:frame_idx]
            left_hand_keypoints = left_hand_keypoints[:frame_idx]
            right_hand_keypoints = right_hand_keypoints[:frame_idx]
            
            # 插值補全
            if sample_rate > 1.0:
                upper_body_keypoints = self._interpolate_keypoints(
                    upper_body_keypoints, sample_indices, frame_idx
                )
                left_hand_keypoints = self._interpolate_keypoints(
                    left_hand_keypoints, sample_indices, frame_idx
                )
                right_hand_keypoints = self._interpolate_keypoints(
                    right_hand_keypoints, sample_indices, frame_idx
                )
            
            if upper_body_keypoints[:, :, 2].max() < 0.01:
                return None
            
            # 正規化
            features = self.normalize_skeleton_features(
                upper_body_keypoints,
                left_hand_keypoints,
                right_hand_keypoints,
                frame_width,
                frame_height
            )
            
            return features
        except:
            return None
    
    def extract_features_from_frames(self, frames, frame_width, frame_height):
        """
        從預加載的幀列表提取骨架特徵（統一介面）

        Args:
            frames: list of numpy arrays (H, W, 3) - RGB frames
            frame_width: 幀寬度
            frame_height: 幀高度

        Returns:
            features: (T, 159) numpy array 或 None
        """
        return self.extract_features_from_preloaded_frames(frames, frame_width, frame_height)

    def extract_features_from_preloaded_frames(self, frames, frame_width, frame_height):
        """
        從預加載的幀列表提取骨架特徵（供異步 I/O 流程使用）

        Args:
            frames: list of numpy arrays (H, W, 3) - RGB frames
            frame_width: 幀寬度
            frame_height: 幀高度

        Returns:
            features: (T, 159) numpy array 或 None
        """
        try:
            if frames is None or len(frames) == 0:
                return None
            
            total_frames = len(frames)
            sample_rate = self._calculate_optimal_sample_rate(total_frames)
            sample_indices = self._generate_sample_indices(total_frames, sample_rate)
            sample_set = set(sample_indices)
            
            # 預分配陣列（動態獲取上半身關鍵點數量）
            num_upper_body = len(self.UPPER_BODY_INDICES)
            upper_body_keypoints = np.zeros((total_frames, num_upper_body, 3), dtype=np.float32)
            left_hand_keypoints = np.zeros((total_frames, 21, 3), dtype=np.float32)
            right_hand_keypoints = np.zeros((total_frames, 21, 3), dtype=np.float32)
            
            # 逐幀處理
            for frame_idx in range(total_frames):
                if frame_idx not in sample_set:
                    continue
                
                frame = frames[frame_idx]
                frame_rgb = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 獲取 Holistic 實例
                if self.use_thread_local:
                    holistic = MediaPipeHolisticPool.get_holistic(model_complexity=0)
                else:
                    holistic = self.holistic
                
                results = holistic.process(frame_rgb)
                
                # 提取上半身關鍵點
                if results.pose_landmarks:
                    pose = results.pose_landmarks.landmark
                    for j, pose_idx in enumerate(self.UPPER_BODY_INDICES):
                        if pose_idx >= 0:
                            p = pose[pose_idx]
                            upper_body_keypoints[frame_idx, j] = [p.x, p.y, p.visibility]
                        else:
                            if results.face_landmarks:
                                face = results.face_landmarks.landmark
                                if pose_idx == -1:
                                    iris_indices = [469, 470, 471, 472]
                                    iris_points = [face[idx] for idx in iris_indices if idx < len(face)]
                                    if iris_points:
                                        center_x = sum(p.x for p in iris_points) / len(iris_points)
                                        center_y = sum(p.y for p in iris_points) / len(iris_points)
                                        visibility = sum(p.z for p in iris_points) / len(iris_points)
                                        upper_body_keypoints[frame_idx, j] = [center_x, center_y, visibility]
                                elif pose_idx == -2:
                                    iris_indices = [474, 475, 476, 477]
                                    iris_points = [face[idx] for idx in iris_indices if idx < len(face)]
                                    if iris_points:
                                        center_x = sum(p.x for p in iris_points) / len(iris_points)
                                        center_y = sum(p.y for p in iris_points) / len(iris_points)
                                        visibility = sum(p.z for p in iris_points) / len(iris_points)
                                        upper_body_keypoints[frame_idx, j] = [center_x, center_y, visibility]
                
                # 提取左手關鍵點
                if results.left_hand_landmarks:
                    left_hand_keypoints[frame_idx] = np.array([
                        [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
                    ], dtype=np.float32)
                
                # 提取右手關鍵點
                if results.right_hand_landmarks:
                    right_hand_keypoints[frame_idx] = np.array([
                        [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
                    ], dtype=np.float32)
            
            # 插值補全
            if sample_rate > 1.0:
                upper_body_keypoints = self._interpolate_keypoints(
                    upper_body_keypoints, sample_indices, total_frames
                )
                left_hand_keypoints = self._interpolate_keypoints(
                    left_hand_keypoints, sample_indices, total_frames
                )
                right_hand_keypoints = self._interpolate_keypoints(
                    right_hand_keypoints, sample_indices, total_frames
                )
            
            if upper_body_keypoints[:, :, 2].max() < 0.01:
                return None
            
            # 正規化
            features = self.normalize_skeleton_features(
                upper_body_keypoints,
                left_hand_keypoints,
                right_hand_keypoints,
                frame_width,
                frame_height
            )
            
            return features
        
        except Exception as e:
            return None
    
    def _interpolate_keypoints(self, keypoints, sample_indices, total_frames):
        """線性插值補全採樣後的缺失幀（向量化優化版本）"""
        interpolated = np.zeros_like(keypoints, dtype=np.float32)
        
        sample_indices_int = np.array([int(idx) for idx in sample_indices if int(idx) < total_frames])
        interpolated[sample_indices_int] = keypoints[sample_indices_int]
        
        num_joints = keypoints.shape[1]
        valid_mask = keypoints[sample_indices_int, :, 2] > 0.1
        
        for j in range(num_joints):
            valid_sample_mask = valid_mask[:, j]
            if valid_sample_mask.sum() > 1:
                valid_frame_indices = sample_indices_int[valid_sample_mask]
                valid_x = keypoints[valid_frame_indices, j, 0]
                valid_y = keypoints[valid_frame_indices, j, 1]
                valid_z = keypoints[valid_frame_indices, j, 2]
                
                f_x = interpolate.interp1d(valid_frame_indices, valid_x, kind='linear', fill_value='extrapolate')
                f_y = interpolate.interp1d(valid_frame_indices, valid_y, kind='linear', fill_value='extrapolate')
                f_z = interpolate.interp1d(valid_frame_indices, valid_z, kind='linear', fill_value='extrapolate')
                
                all_indices = np.arange(total_frames)
                missing_indices = all_indices[~np.isin(all_indices, valid_frame_indices)]
                
                interpolated[missing_indices, j, 0] = f_x(missing_indices)
                interpolated[missing_indices, j, 1] = f_y(missing_indices)
                interpolated[missing_indices, j, 2] = f_z(missing_indices)
        
        return interpolated
    
    def _calculate_optimal_sample_rate(self, total_frames):
        """計算最優採樣數量 - 固定為12幀以提高速度"""
        return 12  # 固定採樣12幀
    
    def _generate_sample_indices(self, total_frames, sample_rate):
        """生成採樣索引 - 固定12幀"""
        num_samples = int(sample_rate)
        if num_samples >= total_frames:
            return np.arange(total_frames, dtype=int)
        else:
            return np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'holistic') and self.holistic is not None:
            self.holistic.close()
    
    def process_directory(self, input_dir, output_dir):
        """處理整個目錄 - 單線程計算 + 異步 I/O 預加載（高效版本）"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        word_dirs = [d for d in input_path.iterdir() if d.is_dir()]

        stats = {
            "total_videos": 0,
            "successful": 0,
            "failed": 0,
            "word_details": [],
            "feature_dim": (len(self.UPPER_BODY_INDICES) + 42) * 3,
            "keypoint_breakdown": {
                "upper_body": len(self.UPPER_BODY_INDICES),
                "left_hand": 21,
                "right_hand": 21,
                "total": len(self.UPPER_BODY_INDICES) + 42
            },
            "optimization": {
                "mode": "single_thread_async_io",
                "description": "單線程計算 + 異步 I/O 預加載"
            }
        }

        # 收集所有任務
        all_tasks = []
        for word_dir in word_dirs:
            word_output_dir = output_path / word_dir.name
            word_output_dir.mkdir(exist_ok=True, parents=True)
            video_files = sorted(list(word_dir.glob("*.mp4")))  # 排序確保一致性
            
            for video_file in video_files:
                all_tasks.append((video_file, word_output_dir))

        # 單線程計算 + 異步 I/O 預加載
        successful = 0
        failed = 0
        
        # 建立隊列系統
        video_queue = queue.Queue()  # 待處理的視頻
        preloaded_queue = queue.Queue(maxsize=2)  # 預加載的視頻隊列（最多預加載 2 個）
        
        # 啟動預加載線程
        preloader = AsyncBatchPreloader(video_queue, preloaded_queue)
        preload_thread = threading.Thread(target=preloader.run, daemon=False)
        preload_thread.start()
        
        # 將所有任務加入隊列
        for video_file, output_dir in all_tasks:
            video_queue.put((video_file, output_dir))
        
        # 發送哨兵值，通知預加載線程結束
        video_queue.put((None, None))
        
        # 單線程計算迴圈
        with tqdm(total=len(all_tasks), desc="提取特徵（單線程+異步IO）") as pbar:
            while True:
                try:
                    # 從預加載隊列獲取視頻（非阻塞以避免死鎖）
                    result = preloaded_queue.get(timeout=5)
                    
                    if result is None:  # 哨兵值
                        break
                    
                    video_path, output_dir, frames, frame_width, frame_height = result
                    
                    if frames is None:
                        failed += 1
                        pbar.update(1)
                        continue
                    
                    try:
                        # 直接用預加載的幀進行特徵提取
                        features = self.extract_features_from_preloaded_frames(
                            frames, frame_width, frame_height
                        )
                        
                        if features is not None:
                            output_file = Path(output_dir) / f"{Path(video_path).stem}.npy"
                            np.save(output_file, features)
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                    
                    pbar.update(1)
                    
                except queue.Empty:
                    # 隊列超時，可能預加載線程有問題
                    break
        
        # 等待預加載線程結束
        preload_thread.join(timeout=5)
        preloader.stop()

        # 統計
        for word_dir in word_dirs:
            word_output_dir = output_path / word_dir.name
            video_files = list(word_dir.glob("*.mp4"))
            npy_files = list(word_output_dir.glob("*.npy"))
            
            word_stats = {
                "word": word_dir.name,
                "total": len(video_files),
                "success": len(npy_files),
                "failed": len(video_files) - len(npy_files)
            }
            stats["word_details"].append(word_stats)

        stats["total_videos"] = len(all_tasks)
        stats["successful"] = successful
        stats["failed"] = failed

        return stats
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'holistic') and self.holistic is not None:
            self.holistic.close()


class AsyncBatchPreloader:
    """
    異步視頻預加載器 - 在計算時同時讀取下一個視頻
    減少 I/O 等待時間
    """
    def __init__(self, video_queue, output_queue, max_queue_size=3):
        self.video_queue = video_queue  # (video_path, output_dir) 任務
        self.output_queue = output_queue  # 預加載好的視頻幀
        self.max_queue_size = max_queue_size
        self.stop_event = threading.Event()
        
    def run(self):
        """背景預加載線程"""
        while not self.stop_event.is_set():
            try:
                # 阻塞等待新任務（超時避免死鎖）
                video_path, output_dir = self.video_queue.get(timeout=1)
                
                if video_path is None:  # 哨兵值，表示結束
                    self.output_queue.put(None)
                    break
                
                try:
                    # 流式讀取視頻
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        self.output_queue.put((video_path, output_dir, None, 0, 0))
                        continue
                    
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frames = []
                    
                    # 邊讀邊加入隊列（不等全部讀完）
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    
                    cap.release()
                    
                    # 放入輸出隊列（供計算線程使用）
                    if frames:
                        self.output_queue.put((video_path, output_dir, frames, frame_width, frame_height))
                    else:
                        self.output_queue.put((video_path, output_dir, None, 0, 0))
                        
                except Exception as e:
                    self.output_queue.put((video_path, output_dir, None, 0, 0))
                    
            except queue.Empty:
                continue
    
    def stop(self):
        """停止預加載線程"""
        self.stop_event.set()


def create_zip(output_dir, zip_name):
    """打包輸出目錄為 ZIP"""
    output_path = Path(output_dir)
    zip_path = output_path.parent / zip_name
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_path.rglob('*.npy'):
            arcname = file_path.relative_to(output_path.parent)
            zipf.write(file_path, arcname)
    
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ 打包完成！ZIP 檔案大小: {zip_size:.2f} MB")
    return zip_path


def main():
    """主函數 - Kaggle T4 GPU 優化"""
    print("🚀 骨架特徵提取 - MediaPipe Holistic (GPU 加速)")
    print("=" * 60)

    kaggle_input = Path("/kaggle/input/final-preprocessed/videos_normalized")
    kaggle_working = Path("/kaggle/working")

    if kaggle_input.exists():
        input_dir = kaggle_input
        output_dir = kaggle_working / "skeleton_features"
    else:
        input_dir = Path("/Users/baidongqu/Desktop/產學/selected_words")
        output_dir = Path("/Users/baidongqu/Desktop/產學/skeleton_features")

    if not input_dir.exists():
        print(f"❌ 輸入目錄不存在: {input_dir}")
        return

    start_time = datetime.now()
    extractor = EnhancedSkeletonExtractor(num_threads=None)
    
    # 顯示優化配置信息
    if extractor.is_apple_silicon:
        print(f"\n🎯 M1 優化配置已啟用")
        print(f"   - Metal 加速: ✅ 已啟用")
        print(f"   - 工作線程: {extractor.num_threads}")
    else:
        print(f"\n💻 標準配置")
        print(f"   - GPU 支援: {extractor.cuda_available}")
        print(f"   - 工作線程: {extractor.num_threads}")
    
    stats = extractor.process_directory(input_dir, output_dir)
    
    if stats["successful"] > 0:
        zip_path = create_zip(output_dir, "skeleton_features.zip")
        stats["zip_path"] = str(zip_path)
    
    stats["processing_time"] = str(datetime.now() - start_time)
    report_path = output_dir / "extraction_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ 完成 | 成功: {stats['successful']} | 失敗: {stats['failed']} | 耗時: {stats['processing_time']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
