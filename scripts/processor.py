"""
手語影片前處理腳本

模組架構：
1. VideoProcessor（核心處理器）- 辨識時也會用到
   - 智能人體檢測和裁切（MediaPipe Pose）
   - 影片標準化（幀數、FPS、解析度）

2. DataAugmentor（數據增強器）- 僅訓練前處理使用
   - 極輕微旋轉、縮放、亮度調整
   - 創建多個增強版本
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import zipfile
import psutil
import threading

# 導入共用工具模組
from utils import MediaPipePosePool, SelfieSegmentationPool


# ==================== MediaPipe 線程池管理 ====================
# 已移至 utils/mediapipe_pool.py
# 使用統一的 MediaPipePosePool 和 SelfieSegmentationPool


# ==================== 數據增強快取機制 ====================

class AugmentationCache:
    """
    增強結果快取 - 避免重複計算相同的增強操作
    """
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def get_key(self, frame_hash, augmentation_type):
        """生成快取鍵"""
        return f"{frame_hash}_{augmentation_type}"
    
    def get(self, frame_hash, augmentation_type):
        """獲取快取的增強結果"""
        key = self.get_key(frame_hash, augmentation_type)
        with self._lock:
            return self._cache.get(key)
    
    def set(self, frame_hash, augmentation_type, result):
        """保存增強結果到快取"""
        key = self.get_key(frame_hash, augmentation_type)
        with self._lock:
            self._cache[key] = result
    
    def clear(self):
        """清理快取"""
        with self._lock:
            self._cache.clear()


# ==================== 數據增強模組（獨立） ====================
class DataAugmentor:
    """
    數據增強器 - 手語辨識最佳化版本
    
    增強策略（保留語意的微調）：
    1. 基礎版本：Original (原始) 和 Mirror (水平翻轉)
    2. 光照增強：亮度調整 (±2%)、對比度調整 (±3%)
    3. 視覺增強：飽和度調整 (±3%)、輕微高斯模糊
    4. 幾何增強：旋轉 (±1°)
    
    組合策略：
    - Original + Original_Brightness + Original_Contrast + Original_Saturation
    - Mirror + Mirror_Brightness + Mirror_Contrast + Mirror_Saturation
    = 共 8 個版本
    """
    
    def __init__(self, cache=None):
        # 增強參數 - 極為保守（針對手語辨識）
        self.augmentation_params = {
            'rotation_range': (-1, 1),              # ±1° 旋轉
            'brightness_range': (0.98, 1.02),      # ±2% 亮度
            'contrast_range': (0.97, 1.03),        # ±3% 對比度
            'saturation_range': (0.97, 1.03),      # ±3% 飽和度
            'blur_sigma': 0.3,                      # 輕微高斯模糊
        }
        self.cache = cache if cache is not None else AugmentationCache()
    
    def _adjust_brightness(self, frame, factor):
        """調整亮度"""
        adjusted = frame.astype(np.float32)
        adjusted = np.clip(adjusted * factor, 0, 255)
        return adjusted.astype(np.uint8)
    
    def _adjust_contrast(self, frame, factor):
        """調整對比度 (相對於中值 128)"""
        adjusted = frame.astype(np.float32)
        adjusted = 128 + (adjusted - 128) * factor
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(np.uint8)
    
    def _adjust_saturation(self, frame):
        """調整飽和度 - 轉換為 HSV 調整 S 通道"""
        # BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        # 調整 S 通道 (±3%)
        saturation_factor = np.random.RandomState(42).choice([0.97, 1.03])
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        # HSV to BGR
        hsv = hsv.astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
    
    def _apply_blur(self, frame, sigma):
        """應用輕微高斯模糊"""
        if sigma > 0:
            kernel_size = 3
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
        return frame
    
    def apply_augmentation(self, frame, augmentation_type=None):
        """
        確定性數據增強（快取版本）
        
        Args:
            frame: 輸入幀 [H, W, 3]
            augmentation_type: 增強類型
                - None: 原始幀
                - 'mirror': 水平翻轉
                - 'brightness': 亮度調整
                - 'contrast': 對比度調整
                - 'saturation': 飽和度調整
                - 'rotation': 旋轉
                - 複合: 'mirror+brightness' 等
        
        Returns:
            增強後的幀 [H, W, 3]
        """
        # 計算幀的雜湊值用於快取
        frame_hash = hash(frame.tobytes())
        
        # 嘗試從快取獲取
        cached_result = self.cache.get(frame_hash, augmentation_type)
        if cached_result is not None:
            return cached_result
        
        augmented = frame.astype(np.uint8)
        
        # 解析複合增強類型（例如 "mirror+brightness"）
        aug_list = augmentation_type.split('+') if augmentation_type else []
        
        for aug_type in aug_list:
            aug_type = aug_type.strip()
            
            if aug_type == 'mirror':
                augmented = cv2.flip(augmented, 1)
            
            elif aug_type == 'brightness':
                factor = np.random.RandomState(42).choice([0.98, 1.02])
                augmented = self._adjust_brightness(augmented, factor)
            
            elif aug_type == 'contrast':
                factor = np.random.RandomState(42).choice([0.97, 1.03])
                augmented = self._adjust_contrast(augmented, factor)
            
            elif aug_type == 'saturation':
                augmented = self._adjust_saturation(augmented)
            
            elif aug_type == 'rotation':
                angle = np.random.RandomState(42).choice([-1, 1])
                h, w = augmented.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h),
                                         borderMode=cv2.BORDER_REFLECT)
            
            elif aug_type == 'blur':
                augmented = self._apply_blur(augmented, self.augmentation_params['blur_sigma'])
        
        result = augmented.astype(np.uint8)
        
        # 保存到快取
        self.cache.set(frame_hash, augmentation_type, result)
        
        return result
    
    def create_augmented_versions(self, frames, output_base_path, save_func):
        """
        為給定的幀序列創建增強版本並保存
        
        增強組合（共 8 個版本）：
        1. Original
        2. Original + Brightness
        3. Original + Contrast
        4. Original + Saturation
        5. Mirror
        6. Mirror + Brightness
        7. Mirror + Contrast
        8. Mirror + Saturation
        
        Args:
            frames: 標準化後的幀列表
            output_base_path: 輸出基礎路徑（不含副檔名）
            save_func: 保存影片的函數，簽名為 save_func(frames, output_path)
        
        Returns:
            成功保存的版本數量
        """
        # 定義增強組合
        augmentation_configs = [
            # Original 系列
            (None, ''),
            ('brightness', '_brightness'),
            ('contrast', '_contrast'),
            ('saturation', '_saturation'),
            # Mirror 系列
            ('mirror', '_mirror'),
            ('mirror+brightness', '_mirror_brightness'),
            ('mirror+contrast', '_mirror_contrast'),
            ('mirror+saturation', '_mirror_saturation'),
        ]
        
        success_count = 0
        output_ext = '.mp4'
        
        for aug_type, suffix in augmentation_configs:
            # 應用增強
            if aug_type is None:
                augmented_frames = frames
            else:
                augmented_frames = [self.apply_augmentation(frame, aug_type) for frame in frames]
            
            # 保存
            output_file = output_base_path + suffix + output_ext
            if save_func(augmented_frames, output_file):
                success_count += 1
        
        return success_count


# ==================== 核心影片處理器 ====================

class VideoProcessor:
    """
    核心影片處理器 - 辨識時也會用到

    功能：
    1. 智能人體檢測和裁切（MediaPipe Pose）
    2. 背景移除（MediaPipe SelfieSegmentation）
    3. 影片標準化（FPS、解析度）
    4. 影片編碼優化（h.264）

    注意：
    - 不包含數據增強功能！數據增強請使用 DataAugmentor
    - 不強制幀數標準化！由呼叫者決定目標幀數
    """
    
    # 標準化參數
    TARGET_FPS = 30         # 目標幀率
    TARGET_WIDTH = 224      # 目標寬度
    TARGET_HEIGHT = 224     # 目標高度
    TARGET_BITRATE = 2000   # 目標比特率 (kbps)
    OUTPUT_EXT = '.mp4'     # 統一輸出格式
    
    # 人體裁切參數
    CROP_PADDING = 0.40     # 邊界框擴展比例（40%）- 大幅增加裁切區域
    MIN_DETECTION_CONFIDENCE = 0.5  # 最低檢測信心度

    # 背景移除參數
    DEFAULT_BG_COLOR = (0, 255, 0)   # 預設背景顏色 (綠幕 BGR 格式)
    SEGMENTATION_MODEL = 1            # SelfieSegmentation 模型選擇 (1=較高品質)
    
    # 邊界優化參數
    MASK_THRESHOLD = 0.5              # 遮罩二值化閾值
    MORPHOLOGY_KERNEL_SIZE = 5        # 形態學操作核心大小
    MORPHOLOGY_ITERATIONS = 2         # 形態學操作迭代次數
    BLUR_KERNEL_SIZE = 5              # 高斯模糊核心大小（平滑邊界）

    def __init__(self, enable_cropping=True, enable_background_removal=False, target_frames=None):
        """
        初始化影片處理器

        Args:
            enable_cropping: 是否啟用智能人體裁切（預設開啟）
            enable_background_removal: 是否啟用背景移除（預設關閉）
            target_frames: 目標幀數（None=不強制標準化，由呼叫者決定）
        """
        # 檢查是否有ffmpeg（用於後處理確保h264編碼）
        self.has_ffmpeg = self._check_ffmpeg()

        # 是否啟用智能裁切
        self.enable_cropping = enable_cropping

        # 是否啟用背景移除
        self.enable_background_removal = enable_background_removal

        # 目標幀數（訓練時使用，辨識時由 SlidingWindowInference 決定）
        self.target_frames = target_frames

        # 第一幀裁切參數（用於固定裁切區域，每個影片獨立）
        self.fixed_crop_params = None

        # 使用線程池管理的 MediaPipe Pose（每個線程獨立實例）
        if self.enable_cropping:
            self.pose_detector = MediaPipePosePool.get_detector(
                min_detection_confidence=self.MIN_DETECTION_CONFIDENCE
            )
        else:
            self.pose_detector = None

        # 使用線程池管理的 SelfieSegmentation（每個線程獨立實例）
        # 注意：不在 __init__ 中初始化，而是在使用時動態獲取以確保線程安全
        # self.segmenter = SelfieSegmentationPool.get_segmenter(model_selection=self.SEGMENTATION_MODEL)

    def _check_ffmpeg(self):
        """檢查系統是否有ffmpeg"""
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def detect_and_crop_person(self, frame, is_first_frame=False):
        """
        檢測並裁切畫面中「最前面」的人的完整上半身
        
        使用 MediaPipe Pose 檢測人體關鍵點，計算包含：
        - 頭部（鼻子、眼睛、耳朵）
        - 上半身（肩膀、手肘、手腕）
        - 完整手臂（確保不裁切）
        
        Args:
            frame: 輸入幀 [H, W, 3] BGR
            is_first_frame: 是否為影片的第一幀（用於固定裁切參數）
        
        Returns:
            cropped_frame: 裁切後的幀（如果檢測失敗返回原始幀）
            success: 是否成功檢測並裁切
        """
        if not self.enable_cropping or self.pose_detector is None:
            return frame, False
        
        h, w = frame.shape[:2]
        
        # 如果已經有固定裁切參數且不是第一幀，直接使用
        if self.fixed_crop_params is not None and not is_first_frame:
            x_min, y_min, x_max, y_max = self.fixed_crop_params
            # 確保裁切區域在當前幀的範圍內
            x_min = max(0, min(x_min, w-1))
            x_max = max(x_min+1, min(x_max, w))
            y_min = max(0, min(y_min, h-1))
            y_max = max(y_min+1, min(y_max, h))
            
            cropped = frame[y_min:y_max, x_min:x_max]
            return cropped if cropped.size > 0 else frame, cropped.size > 0
        
        # 轉換為 RGB（MediaPipe 需要 RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 執行姿態檢測
        results = self.pose_detector.process(frame_rgb)
        
        # 如果沒有檢測到人體，返回原始幀
        if not results.pose_landmarks:
            return frame, False
        
        # 提取關鍵點
        landmarks = results.pose_landmarks.landmark
        
        # 定義需要包含的關鍵點（主要追焦頭部 + 肩膀，避免手臂干擾）
        # MediaPipe Pose 關鍵點索引：
        # 0: 鼻子, 1-2: 眼睛, 3-4: 耳朵
        # 11-12: 肩膀, 13-14: 手肘, 15-16: 手腕, 17-22: 手指
        head_and_shoulders_indices = [
            0,   # 鼻子（頭部中心）
            1, 2,  # 左右眼睛
            3, 4,  # 左右耳朵
            11, 12,  # 左右肩膀（提供上半身穩定性）
        ]
        
        # 計算邊界框
        x_coords = []
        y_coords = []
        
        for idx in head_and_shoulders_indices:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                # 檢查關鍵點可見性（visibility > 0.5 表示可見）
                if landmark.visibility > 0.5:
                    x_coords.append(landmark.x * w)
                    y_coords.append(landmark.y * h)
        
        # 如果檢測到的關鍵點太少，返回原始幀
        if len(x_coords) < 4:
            return frame, False
        
        # 計算邊界框
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        # 添加 padding（確保不裁切到手臂）
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        padding_x = int(bbox_width * self.CROP_PADDING)
        padding_y = int(bbox_height * self.CROP_PADDING)
        
        x_min = max(0, x_min - padding_x)
        x_max = min(w, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(h, y_max + padding_y)
        
        # 固定裁切區域大小，避免視覺上的放大縮小
        # 使用固定的邊長（取原始邊界框的最大值作為基準，並增加60%以獲得更大區域）
        base_size = max(x_max - x_min, y_max - y_min)
        fixed_size = int(base_size * 1.6)  # 增加60%讓裁切框更大
        
        # 以檢測到的中心為基準，創建固定大小的裁切區域
        # 稍微往下調整中心點，避免裁切框偏上
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # 往下調整中心點 10%，讓裁切框包含更多下方區域
        center_y = int(center_y * 1.1)
        
        half_size = fixed_size // 2
        x_min = max(0, center_x - half_size)
        x_max = min(w, center_x + half_size)
        y_min = max(0, center_y - half_size)
        y_max = min(h, center_y + half_size)
        
        # 如果裁切區域小於固定大小，嘗試居中擴展
        if x_max - x_min < fixed_size:
            expand = (fixed_size - (x_max - x_min)) // 2
            x_min = max(0, x_min - expand)
            x_max = min(w, x_max + expand)
        
        if y_max - y_min < fixed_size:
            expand = (fixed_size - (y_max - y_min)) // 2
            y_min = max(0, y_min - expand)
            y_max = min(h, y_max + expand)
        
        # 如果是第一幀，儲存裁切參數
        if is_first_frame:
            self.fixed_crop_params = (x_min, y_min, x_max, y_max)
        
        # 裁切
        cropped = frame[y_min:y_max, x_min:x_max]
        
        # 檢查裁切結果是否有效
        if cropped.size == 0:
            return frame, False
        
        return cropped, True

    def apply_background_removal(self, frame, bg_color=None):
        """
        應用背景移除到單一幀（優化版：邊界平滑處理 + 線程安全）

        優化策略：
        1. 提高輸入解析度（保持原始解析度處理）
        2. 形態學操作平滑邊界（膨脹+侵蝕）
        3. 高斯模糊減少鋸齒
        4. 綠幕背景便於觀察調整
        5. 線程安全：處理 MediaPipe 時間戳衝突

        Args:
            frame: 輸入幀 [H, W, 3] BGR
            bg_color: 背景顏色 (B, G, R)，預設為綠幕

        Returns:
            處理後的幀 [H, W, 3] BGR（前景保持，背景為綠幕）
        """
        if not self.enable_background_removal:
            return frame

        if bg_color is None:
            bg_color = self.DEFAULT_BG_COLOR

        # 線程安全：確保使用當前線程的獨立實例
        segmenter = SelfieSegmentationPool.get_segmenter(model_selection=self.SEGMENTATION_MODEL)

        try:
            h, w = frame.shape[:2]

            # 步驟1: 轉換為 RGB（MediaPipe 需要 RGB 格式）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 步驟2: 進行分割（使用原始解析度提升精度）
            results = segmenter.process(frame_rgb)
            mask = results.segmentation_mask

            if mask is not None:
                # 確保遮罩和幀有相同的尺寸
                mask = cv2.resize(mask, (w, h))

                # 步驟3: 二值化遮罩（提高對比度）
                mask_binary = (mask > self.MASK_THRESHOLD).astype(np.uint8)

                # 步驟4: 形態學操作平滑邊界
                # 4.1 膨脹操作：填補人體內部的小孔洞
                kernel = np.ones((self.MORPHOLOGY_KERNEL_SIZE, self.MORPHOLOGY_KERNEL_SIZE), np.uint8)
                mask_dilated = cv2.dilate(mask_binary, kernel, iterations=self.MORPHOLOGY_ITERATIONS)

                # 4.2 侵蝕操作：縮小邊界，移除噪點
                mask_eroded = cv2.erode(mask_dilated, kernel, iterations=self.MORPHOLOGY_ITERATIONS)

                # 步驟5: 高斯模糊平滑邊界（減少鋸齒）
                mask_smooth = cv2.GaussianBlur(mask_eroded.astype(np.float32),
                                              (self.BLUR_KERNEL_SIZE, self.BLUR_KERNEL_SIZE), 0)

                # 步驟6: 創建綠幕背景
                green_background = np.full((h, w, 3), bg_color, dtype=np.uint8)

                # 步驟7: 根據平滑遮罩合成最終圖像
                # 使用加權混合獲得平滑過渡
                mask_3channel = mask_smooth[:, :, np.newaxis]
                output_frame = (frame * mask_3channel + green_background * (1 - mask_3channel)).astype(np.uint8)

                return output_frame
            else:
                # 如果分割失敗，返回原始幀
                return frame

        except Exception as e:
            error_msg = str(e)
            # 檢查是否是 MediaPipe 時間戳衝突錯誤
            if "timestamp mismatch" in error_msg.lower() or "packet timestamp" in error_msg.lower():
                print(f"⚠️  MediaPipe 時間戳衝突，嘗試重新初始化實例...")
                try:
                    # 清理當前線程實例並重新獲取
                    SelfieSegmentationPool.close_thread_segmenter()
                    segmenter = SelfieSegmentationPool.get_segmenter(model_selection=self.SEGMENTATION_MODEL)

                    # 重試一次
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = segmenter.process(frame_rgb)
                    mask = results.segmentation_mask

                    if mask is not None:
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_binary = (mask > self.MASK_THRESHOLD).astype(np.uint8)
                        kernel = np.ones((self.MORPHOLOGY_KERNEL_SIZE, self.MORPHOLOGY_KERNEL_SIZE), np.uint8)
                        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=self.MORPHOLOGY_ITERATIONS)
                        mask_eroded = cv2.erode(mask_dilated, kernel, iterations=self.MORPHOLOGY_ITERATIONS)
                        mask_smooth = cv2.GaussianBlur(mask_eroded.astype(np.float32),
                                                      (self.BLUR_KERNEL_SIZE, self.BLUR_KERNEL_SIZE), 0)
                        green_background = np.full(frame.shape, bg_color, dtype=np.uint8)
                        mask_3channel = mask_smooth[:, :, np.newaxis]
                        output_frame = (frame * mask_3channel + green_background * (1 - mask_3channel)).astype(np.uint8)
                        print("✅ 重試成功")
                        return output_frame

                except Exception as retry_e:
                    print(f"❌ 重試也失敗: {str(retry_e)}")

            print(f"背景移除處理失敗: {error_msg}")
            import traceback
            traceback.print_exc()
            return frame

    def normalize_frames(self, frames, target_count):
        """
        智能幀數標準化
        - 如果原始幀數 > 目標幀數：均勻採樣
        - 如果原始幀數 < 目標幀數：線性插值
        
        Args:
            frames: 原始幀列表
            target_count: 目標幀數
            
        Returns:
            標準化後的幀列表
        """
        original_count = len(frames)
        
        if original_count == target_count:
            return frames
        
        # 生成採樣索引
        indices = np.linspace(0, original_count - 1, target_count)
        
        normalized_frames = []
        for idx in indices:
            # 如果是整數索引，直接取幀
            if idx == int(idx):
                normalized_frames.append(frames[int(idx)])
            else:
                # 線性插值
                lower_idx = int(np.floor(idx))
                upper_idx = int(np.ceil(idx))
                weight = idx - lower_idx
                
                # 混合兩幀
                lower_frame = frames[lower_idx].astype(np.float32)
                upper_frame = frames[upper_idx].astype(np.float32)
                interpolated = (1 - weight) * lower_frame + weight * upper_frame
                normalized_frames.append(interpolated.astype(np.uint8))
        
        return normalized_frames
    
    def process_single_frame(self, frame, is_first_frame=False):
        """
        處理單一幀：裁切 + 背景移除 + 解析度標準化
        
        Args:
            frame: 輸入幀
            is_first_frame: 是否為第一幀（用於固定裁切參數）
        
        Returns:
            處理後的幀
        """
        # 1. 智能裁切人體（如果啟用）
        if self.enable_cropping:
            frame, _ = self.detect_and_crop_person(frame, is_first_frame)
        
        # 2. 背景移除（如果啟用）
        if self.enable_background_removal:
            frame = self.apply_background_removal(frame)
        
        # 3. 解析度標準化
        frame = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
        
        return frame

    def _convert_to_h264(self, input_video, output_video):
        """使用ffmpeg轉換為h264編碼，並設定比特率"""
        import subprocess
        
        cmd = [
            'ffmpeg',
            '-i', input_video,
            '-c:v', 'libx264',              # h264編碼
            '-b:v', f'{self.TARGET_BITRATE}k',  # 比特率
            '-preset', 'medium',            # 編碼速度
            '-movflags', '+faststart',      # 優化串流
            '-y',                           # 覆蓋輸出
            output_video
        ]
        
        try:
            subprocess.run(cmd, 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _save_video_frames(self, frames, output_path):
        """
        將幀序列保存為影片文件

        Args:
            frames: 幀列表
            output_path: 輸出影片路徑

        Returns:
            保存是否成功
        """
        try:
            temp_output = output_path + '.temp.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(
                temp_output,
                fourcc,
                self.TARGET_FPS,
                (self.TARGET_WIDTH, self.TARGET_HEIGHT)
            )

            if not out.isOpened():
                print(f"❌ 無法創建輸出影片: {output_path}")
                return False

            for frame in frames:
                out.write(frame)

            out.release()

            # 使用ffmpeg轉換為標準h264編碼（如果有ffmpeg）
            if self.has_ffmpeg:
                if self._convert_to_h264(temp_output, output_path):
                    # 轉換成功，刪除臨時檔案
                    os.remove(temp_output)
                else:
                    # 轉換失敗，使用臨時檔案作為最終輸出
                    print(f"⚠️  ffmpeg轉換失敗，使用mp4v編碼: {os.path.basename(output_path)}")
                    os.rename(temp_output, output_path)
            else:
                # 沒有ffmpeg，直接使用mp4v編碼
                os.rename(temp_output, output_path)

            # 驗證輸出
            verify_cap = cv2.VideoCapture(output_path)
            verify_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            verify_cap.release()

            # 檢查是否符合規格（允許1幀的誤差，如果有設定目標幀數）
            if self.target_frames is not None and abs(verify_frames - self.target_frames) > 1:
                print(f"⚠️  警告: {os.path.basename(output_path)} 幀數不符 (期望{self.target_frames}, 實際{verify_frames})")

            return True

        except Exception as e:
            print(f"❌ 保存影片時出錯: {output_path}, 錯誤: {str(e)}")
            return False

    def process_video(self, input_path, output_path, augmentor=None, target_frames=None):
        """
        處理單一影片：標準化 + 可選的數據增強
        
        ⚡ 優化後的處理流程（2025最佳實踐）：
        1. 讀取影片（流式處理，減少記憶體佔用）
        2. 智能裁切人體（在原始解析度下進行，更精確）
        3. 背景移除（在裁切後的較小圖像上處理，更快速）
        4. 解析度標準化（224x224）
        5. 幀數標準化（在小圖上插值，減少計算量）
        6. 數據增強（可選，僅在最終尺寸上進行）
        
        如果提供 augmentor：
        - 創建8個增強版本（原始、旋轉、縮放、亮度 × 無反轉/反轉）
        
        如果不提供 augmentor：
        - 只輸出1個標準化版本
        
        Args:
            input_path: 輸入影片路徑
            output_path: 輸出影片路徑
            augmentor: DataAugmentor 實例（可選，訓練時使用）
            target_frames: 目標幀數（可選，覆蓋 __init__ 設定）
        
        Returns:
            處理是否成功
        """
        try:
            # 確保輸出副檔名為.mp4
            output_path_base = os.path.splitext(output_path)[0]

            # 確保輸出目錄存在
            output_dir = os.path.dirname(output_path)
            if output_dir:  # 只有當有目錄部分時才創建
                os.makedirs(output_dir, exist_ok=True)

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"❌ 無法開啟影片: {input_path}")
                return False

            # 讀取所有幀
            # 注意：當前實作一次性讀取所有幀到記憶體
            # 優點：簡化處理流程，方便幀數標準化
            # 缺點：大影片可能佔用較多記憶體
            # 
            # 流式處理優化方案（未來可選）：
            # 1. 逐幀讀取 → 裁切 → 背景移除 → resize → 暫存
            # 2. 完成後再進行幀數標準化和數據增強
            # 3. 預期記憶體減少 50-70%，但代碼複雜度增加
            original_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 修正影片方向：如果幀是豎向的（高度>寬度），旋轉為橫向
                if frame.shape[0] > frame.shape[1]:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 順時針旋轉90度
                original_frames.append(frame)

            cap.release()

            if len(original_frames) == 0:
                print(f"❌ 影片無有效幀: {input_path}")
                return False

            # 重置固定裁切參數（為每個新影片重置，避免線程間干擾）
            self.fixed_crop_params = None

            # ========== 優化流程開始（異步並行處理） ==========
            
            # 步驟1: 處理第一幀以設置固定裁切參數
            if len(original_frames) > 0:
                first_processed = self.process_single_frame(original_frames[0], is_first_frame=True)
                processed_frames = [first_processed]
                
                # 步驟2: 並行處理剩餘幀（使用 ThreadPoolExecutor 避免阻塞）
                if len(original_frames) > 1:
                    with ThreadPoolExecutor(max_workers=min(4, len(original_frames)-1)) as executor:
                        # 並行處理剩餘幀
                        remaining_processed = list(executor.map(
                            lambda f: self.process_single_frame(f, is_first_frame=False),
                            original_frames[1:]
                        ))
                        processed_frames.extend(remaining_processed)
            else:
                processed_frames = []

            # 顯示處理統計（如果有裁切）
            if self.enable_cropping and len(processed_frames) > 0:
                # 簡單估計裁切成功率（實際上在並行處理中難以精確統計）
                print(f"✅ 幀處理完成: {len(processed_frames)} 幀已並行處理")

            # 步驟3: 幀數標準化（在小圖上進行插值，減少計算量）
            target = target_frames if target_frames is not None else self.target_frames
            if target is not None and len(processed_frames) != target:
                normalized_frames = self.normalize_frames(processed_frames, target)
            else:
                normalized_frames = processed_frames

            # 步驟5: 數據增強（可選，僅在最終尺寸上進行）
            if augmentor is not None:
                # 使用 DataAugmentor 創建8個增強版本（Original + Mirror 各配以光照增強）
                success_count = augmentor.create_augmented_versions(
                    normalized_frames,
                    output_path_base,
                    self._save_video_frames
                )
                return success_count == 8  # 所有8個版本都必須成功
            else:
                # 不使用數據增強，只輸出標準化版本
                output_file = output_path_base + self.OUTPUT_EXT
                return self._save_video_frames(normalized_frames, output_file)

        except Exception as e:
            print(f"❌ 處理影片時出錯: {input_path}, 錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def process_directory(self, input_dir, output_dir, max_workers=None, augmentor=None, use_multiprocessing=True):
        """
        處理整個目錄下的所有影片
        
        ⚡ 2025 優化版本：
        1. 使用 ProcessPoolExecutor 避免 Python GIL 限制（預設）
        2. 每個進程獨立的 VideoProcessor 實例
        3. 根據 CPU 核心數自動調整工作進程數
        4. 支援回退到 ThreadPoolExecutor（相容性模式）
        
        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            max_workers: 工作進程數（None=自動檢測）
            augmentor: DataAugmentor 實例（可選，訓練時使用）
            use_multiprocessing: 是否使用多進程（True=ProcessPool, False=ThreadPool）
        """
        if not os.path.exists(input_dir):
            print(f"❌ 輸入目錄不存在: {input_dir}")
            return
            
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # CPU 硬體優化：自動檢測 CPU 核心數
        physical_cores = psutil.cpu_count(logical=False)
        if max_workers is None:
            if use_multiprocessing:
                # ProcessPoolExecutor: 可以使用更多進程（無 GIL 限制）
                max_workers = max(2, physical_cores - 1)  # 保留 1 核給系統
            else:
                # ThreadPoolExecutor: 限制較少線程（有 GIL 限制）
                max_workers = max(2, int((physical_cores - 1) * 0.75))
        
        # 限制最大工作數，避免記憶體過度消耗
        if use_multiprocessing:
            max_workers = max(1, min(max_workers, physical_cores))  # 進程數不超過核心數
        else:
            max_workers = max(1, min(max_workers, 3))  # 線程數最多 3 個
        
        print(f"🚀 使用 {'多進程 (ProcessPool)' if use_multiprocessing else '多線程 (ThreadPool)'} 模式")
        print(f"   工作進程/線程數: {max_workers}/{physical_cores} 核心")
        
        # 獲取所有影片文件
        video_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
                    rel_dir = os.path.relpath(root, input_dir)
                    src_path = os.path.join(root, file)
                    
                    # 保持相同的目錄結構
                    if rel_dir == '.':
                        dest_dir = output_dir
                    else:
                        dest_dir = os.path.join(output_dir, rel_dir)
                    
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # 統一輸出檔名為.mp4
                    file_base = os.path.splitext(file)[0]
                    dest_file = file_base + self.OUTPUT_EXT
                    dest_path = os.path.join(dest_dir, dest_file)
                    
                    video_files.append((src_path, dest_path, self.enable_cropping, 
                                      self.enable_background_removal, self.target_frames, 
                                      augmentor is not None))
        
        # 選擇執行器
        ExecutorClass = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
        
        # 使用多進程/多線程處理影片
        with ExecutorClass(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(_process_video_worker_static, video_files),
                total=len(video_files), 
                desc="🎬 處理影片"
            ))
        
        success_count = sum(results)
        total_count = len(video_files)
        print(f"\n✅ 處理完成：成功 {success_count}/{total_count} 個影片")
        if success_count < total_count:
            print(f"⚠️  失敗 {total_count - success_count} 個影片")

    # ========== 已移除 process_directory_async() ==========
    # 原因：假異步處理無法提供真正的性能提升
    # 1. cv2.VideoCapture 是阻塞式 I/O，無法異步
    # 2. MediaPipe 推理本身是同步操作
    # 3. asyncio 只是包裝了 ThreadPoolExecutor，沒有額外收益
    # 
    # 建議：使用 process_directory(use_multiprocessing=True) 獲得更好性能
    # ProcessPoolExecutor 避免 GIL 限制，真正實現並行處理


# ==================== 多進程工作函數（模組級別） ====================

def _process_video_worker_static(args):
    """
    靜態工作函數 - 用於 ProcessPoolExecutor（必須是模組級別函數）
    
    這個函數必須在模組級別定義，因為 ProcessPoolExecutor 需要序列化函數。
    類方法無法被 pickle 序列化，所以需要使用靜態函數。
    
    Args:
        args: (src_path, dest_path, enable_cropping, enable_background_removal, 
               target_frames, use_augmentor) 元組
    
    Returns:
        處理是否成功
    """
    src_path, dest_path, enable_cropping, enable_background_removal, target_frames, use_augmentor = args
    
    try:
        # 每個進程創建獨立的 VideoProcessor 實例
        processor = VideoProcessor(
            enable_cropping=enable_cropping,
            enable_background_removal=enable_background_removal,
            target_frames=target_frames
        )
        
        # 如果需要數據增強，創建 DataAugmentor
        if use_augmentor:
            augmentor = DataAugmentor()
            result = processor.process_video(src_path, dest_path, augmentor=augmentor)
        else:
            result = processor.process_video(src_path, dest_path, augmentor=None)
        
        # 清理資源（多進程環境下每個進程獨立，無需擔心線程衝突）
        if enable_cropping:
            MediaPipePosePool.close_thread_detector()
        if enable_background_removal:
            SelfieSegmentationPool.close_thread_segmenter()
        
        return result
    
    except Exception as e:
        print(f"❌ 進程處理失敗 {os.path.basename(src_path)}: {str(e)}")
        return False


# ==================== 辨識前處理函數 ====================

def process_for_inference(input_video, output_video, enable_cropping=True, target_frames=None):
    """
    辨識前處理 - 智能裁切 + 背景移除 + 標準化（不含數據增強）
    
    用於辨識時的影片前處理：
    1. 智能裁切人體（可選）
    2. 背景移除（啟用）
    3. 幀數標準化（預設80幀）
    4. 解析度標準化（224x224）
    
    不包含數據增強！僅輸出單個標準化版本
    
    Args:
        input_video: 輸入影片路徑
        output_video: 輸出影片路徑
        enable_cropping: 是否啟用智能裁切（預設開啟）
        target_frames: 目標幀數（預設80幀）
    
    Returns:
        處理是否成功
    
    使用範例：
        >>> from processor import process_for_inference
        >>> success = process_for_inference('input.mp4', 'output.mp4')
    """
    print("=" * 70)
    print("手語影片前處理：智能裁切 + 背景移除 + 標準化（辨識用）")
    print("=" * 70)
    
    # 初始化處理器（啟用背景移除，設定目標幀數）
    processor = VideoProcessor(
        enable_cropping=enable_cropping, 
        enable_background_removal=True,  # 啟用背景移除
        target_frames=target_frames
    )
    
    # 處理影片（augmentor=None，只輸出標準化版本）
    success = processor.process_video(input_video, output_video, augmentor=None, target_frames=target_frames)
    
    if success:
        print(f"✅ 處理完成: {output_video}")
    else:
        print(f"❌ 處理失敗: {input_video}")
    
    return success


def main():
    """
    主程式 - 多進程訓練前處理（包含背景移除 + 數據增強）
    
    ⚡ 2025 優化版本：
    1. 智能人體裁切（原始解析度下更精確）
    2. 背景移除（裁切後更快速）
    3. 數據增強（8個版本）
    4. 多進程處理（避免 GIL，真正並行）
    
    如果只需要標準化（辨識時使用），請參考 process_for_inference()
    """
    print("=" * 70)
    print("手語影片前處理：多進程處理 + 智能裁切 + 背景移除 + 數據增強")
    print("=" * 70)

    # 檢測是否在 Kaggle 環境中
    is_kaggle = os.path.exists('/kaggle')

    if is_kaggle:
        # Kaggle 環境配置
        input_dir = "/kaggle/input/augment/augmented_videos"
        output_dir = "/kaggle/working/videos_processed"
    else:
        # 本地環境配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 使用 bai_dataset 作為輸入
        input_dir = os.path.join(os.path.dirname(current_dir), "Final_MVP/bai_dataset2")
        output_dir = os.path.join(os.path.dirname(current_dir), "Final_MVP/videos_processed")

    # 初始化處理器（啟用智能裁切 + 背景移除）
    processor = VideoProcessor(
        enable_cropping=True,
        enable_background_removal=True,  # ⚡ 啟用背景移除（綠幕效果）
        target_frames=80  # 設定目標幀數為80幀
    )

    # 初始化數據增強器（訓練時使用）
    augmentor = DataAugmentor()

    # ⚡ 使用多進程處理（避免 Python GIL）
    processor.process_directory(
        input_dir, 
        output_dir, 
        augmentor=augmentor,
        use_multiprocessing=True  # 啟用多進程模式
    )

    # 打包輸出目錄為zip檔案
    if is_kaggle:
        zip_path = "/kaggle/working/videos_processed.zip"
    else:
        zip_path = os.path.join(os.path.dirname(current_dir), "videos_processed.zip")

    try:
        print("\n📦 打包輸出檔案...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_kaggle:
                        arcname = os.path.relpath(file_path, "/kaggle/working")
                    else:
                        arcname = os.path.relpath(file_path, os.path.dirname(current_dir))
                    zipf.write(file_path, arcname)

        print(f"✅ 打包完成: {zip_path}")
    except Exception as e:
        print(f"❌ 打包失敗: {str(e)}")

    print("\n" + "=" * 70)
    print("多進程前處理完成！")
    print("=" * 70)

    # 清理全域資源
    MediaPipePosePool.close_all()
    SelfieSegmentationPool.close_all()


if __name__ == "__main__":
    # 直接使用多進程處理（移除 asyncio）
    main()
