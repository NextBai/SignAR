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
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import zipfile
import psutil
import mediapipe as mp


# ==================== 數據增強模組（獨立） ====================
class DataAugmentor:
    """
    數據增強器 - 僅用於訓練前處理
    
    功能：
    - 極輕微旋轉 (±2°)
    - 極輕微縮放 (0.98-1.02x)
    - 極輕微亮度調整 (±5%)
    - 水平翻轉
    
    注意：辨識時不需要使用此模組！
    """
    
    def __init__(self):
        # 數據增強參數 - 極為保守
        self.augmentation_params = {
            'rotation_range': (-2, 2),      # 極輕微旋轉角度範圍（度）
            'scale_range': (0.98, 1.02),    # 極輕微縮放範圍
            'brightness_range': (0.95, 1.05), # 極輕微亮度調整範圍
        }
    
    def apply_augmentation(self, frame, augmentation_type=None):
        """
        確定性數據增強
        
        Args:
            frame: 輸入幀 [H, W, 3]
            augmentation_type: 增強類型 ('rotation', 'scale', 'brightness', 'flip', None)
        
        Returns:
            增強後的幀 [H, W, 3]
        """
        augmented = frame.astype(np.float32)

        if augmentation_type == 'rotation':
            # 確定性旋轉 (±2°)
            angle = np.random.RandomState(42).choice([-2, -1, 1, 2])
            h, w = augmented.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h),
                                     borderMode=cv2.BORDER_REFLECT)

        elif augmentation_type == 'scale':
            # 確定性縮放 (0.98 或 1.02)
            scale = np.random.RandomState(42).choice([0.98, 1.02])
            h, w = augmented.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)

            scaled = cv2.resize(augmented, (new_w, new_h))

            # 居中放置縮放後的圖像
            result = np.zeros_like(augmented)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2

            if scale > 1.0:  # 放大：裁剪中心區域
                src_start_x = (new_w - w) // 2
                src_start_y = (new_h - h) // 2
                result[:, :] = scaled[src_start_y:src_start_y+h, src_start_x:src_start_x+w]
            else:  # 縮小：居中放置
                end_x = start_x + new_w
                end_y = start_y + new_h
                result[start_y:end_y, start_x:end_x] = scaled

            augmented = result

        elif augmentation_type == 'brightness':
            # 確定性亮度調整 (±5%)
            brightness_factor = np.random.RandomState(42).choice([0.95, 1.05])
            augmented = np.clip(augmented * brightness_factor, 0, 255)

        elif augmentation_type == 'flip':
            # 水平反轉
            augmented = cv2.flip(augmented, 1)

        return augmented.astype(np.uint8)
    
    def create_augmented_versions(self, frames, output_base_path, save_func):
        """
        為給定的幀序列創建8個增強版本並保存
        
        Args:
            frames: 標準化後的幀列表
            output_base_path: 輸出基礎路徑（不含副檔名）
            save_func: 保存影片的函數，簽名為 save_func(frames, output_path)
        
        Returns:
            成功保存的版本數量
        """
        base_augmentation_types = [None, 'rotation', 'scale', 'brightness']
        base_suffixes = ['', '_rotation', '_scale', '_brightness']
        
        success_count = 0
        output_ext = '.mp4'
        
        for base_aug, base_suffix in zip(base_augmentation_types, base_suffixes):
            # 創建基本增強版本
            if base_aug is None:
                base_frames = frames
            else:
                base_frames = [self.apply_augmentation(frame, base_aug) for frame in frames]
            
            # 創建不反轉版本
            output_file = output_base_path + base_suffix + output_ext
            if save_func(base_frames, output_file):
                success_count += 1
            
            # 創建反轉版本
            flip_frames = [self.apply_augmentation(frame, 'flip') for frame in base_frames]
            flip_suffix = base_suffix + '_flip' if base_suffix else '_flip'
            output_file_flip = output_base_path + flip_suffix + output_ext
            if save_func(flip_frames, output_file_flip):
                success_count += 1
        
        return success_count


# ==================== 核心影片處理器 ====================

class VideoProcessor:
    """
    核心影片處理器 - 辨識時也會用到
    
    功能：
    1. 智能人體檢測和裁切（MediaPipe Pose）
    2. 影片標準化（幀數、FPS、解析度）
    3. 影片編碼優化（h.264）
    
    注意：不包含數據增強功能！數據增強請使用 DataAugmentor
    """
    
    # 標準化參數
    TARGET_FRAMES = 80      # 目標幀數
    TARGET_FPS = 30         # 目標幀率
    TARGET_WIDTH = 224      # 目標寬度
    TARGET_HEIGHT = 224     # 目標高度
    TARGET_BITRATE = 2000   # 目標比特率 (kbps)
    OUTPUT_EXT = '.mp4'     # 統一輸出格式
    
    # 人體裁切參數
    CROP_PADDING = 0.15     # 邊界框擴展比例（15%）
    MIN_DETECTION_CONFIDENCE = 0.5  # 最低檢測信心度
    
    def __init__(self, enable_cropping=True):
        """
        初始化影片處理器
        
        Args:
            enable_cropping: 是否啟用智能人體裁切（預設開啟）
        """
        # 檢查是否有ffmpeg（用於後處理確保h264編碼）
        self.has_ffmpeg = self._check_ffmpeg()
        
        # 是否啟用智能裁切
        self.enable_cropping = enable_cropping
        
        # 初始化 MediaPipe Pose（延遲初始化）
        self.mp_pose = None
        self.pose_detector = None
        
        if self.enable_cropping:
            print("🔧 初始化 MediaPipe Pose 檢測器...")
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
                min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Pose 已初始化")
        
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
    
    def detect_and_crop_person(self, frame):
        """
        檢測並裁切畫面中「最前面」的人的完整上半身
        
        使用 MediaPipe Pose 檢測人體關鍵點，計算包含：
        - 頭部（鼻子、眼睛、耳朵）
        - 上半身（肩膀、手肘、手腕）
        - 完整手臂（確保不裁切）
        
        Args:
            frame: 輸入幀 [H, W, 3] BGR
        
        Returns:
            cropped_frame: 裁切後的幀（如果檢測失敗返回原始幀）
            success: 是否成功檢測並裁切
        """
        if not self.enable_cropping or self.pose_detector is None:
            return frame, False
        
        h, w = frame.shape[:2]
        
        # 轉換為 RGB（MediaPipe 需要 RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 執行姿態檢測
        results = self.pose_detector.process(frame_rgb)
        
        # 如果沒有檢測到人體，返回原始幀
        if not results.pose_landmarks:
            return frame, False
        
        # 提取關鍵點
        landmarks = results.pose_landmarks.landmark
        
        # 定義需要包含的關鍵點（上半身 + 手臂）
        # MediaPipe Pose 關鍵點索引：
        # 0: 鼻子, 1-2: 眼睛, 3-4: 耳朵
        # 11-12: 肩膀, 13-14: 手肘, 15-16: 手腕, 17-22: 手指（可選）
        upper_body_indices = [
            0,   # 鼻子（頭部頂部參考）
            1, 2, 3, 4,  # 眼睛、耳朵
            11, 12,  # 左右肩膀
            13, 14,  # 左右手肘
            15, 16,  # 左右手腕
            17, 18, 19, 20, 21, 22  # 手指（確保完整手部）
        ]
        
        # 計算邊界框
        x_coords = []
        y_coords = []
        
        for idx in upper_body_indices:
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
        
        # 確保裁切區域是正方形（避免變形）
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        
        if crop_width > crop_height:
            # 寬度大於高度，擴展高度
            diff = crop_width - crop_height
            y_min = max(0, y_min - diff // 2)
            y_max = min(h, y_max + diff // 2)
        else:
            # 高度大於寬度，擴展寬度
            diff = crop_height - crop_width
            x_min = max(0, x_min - diff // 2)
            x_max = min(w, x_max + diff // 2)
        
        # 裁切
        cropped = frame[y_min:y_max, x_min:x_max]
        
        # 檢查裁切結果是否有效
        if cropped.size == 0:
            return frame, False
        
        return cropped, True
    
    def __del__(self):
        """清理資源"""
        if self.pose_detector is not None:
            self.pose_detector.close()
    
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
    
    def process_and_crop_frames(self, frames):
        """
        處理幀序列：智能裁切 + 解析度標準化
        
        注意：此函數不包含數據增強！僅用於核心處理
        
        Args:
            frames: 輸入幀列表
        
        Returns:
            處理後的幀列表（標準化到 224x224）
        """
        processed_frames = []
        
        for frame in frames:
            # 1. 智能裁切人體（如果啟用）
            if self.enable_cropping:
                frame_cropped, success = self.detect_and_crop_person(frame)
                if success:
                    frame = frame_cropped
            
            # 2. 解析度標準化
            frame_resized = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
            processed_frames.append(frame_resized)
        
        return processed_frames

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

            # 檢查是否符合規格（允許1幀的誤差）
            if abs(verify_frames - self.TARGET_FRAMES) > 1:
                print(f"⚠️  警告: {os.path.basename(output_path)} 幀數不符 (期望{self.TARGET_FRAMES}, 實際{verify_frames})")

            return True

        except Exception as e:
            print(f"❌ 保存影片時出錯: {output_path}, 錯誤: {str(e)}")
            return False

    def process_video(self, input_path, output_path, augmentor=None):
        """
        處理單一影片：標準化 + 可選的數據增強
        
        核心處理流程：
        1. 幀數標準化（80幀）
        2. 智能裁切人體（可選）
        3. 解析度標準化（224x224）
        4. 數據增強（可選，需要提供 augmentor）
        
        如果提供 augmentor：
        - 創建8個增強版本（原始、旋轉、縮放、亮度 × 無反轉/反轉）
        
        如果不提供 augmentor：
        - 只輸出1個標準化版本
        
        Args:
            input_path: 輸入影片路徑
            output_path: 輸出影片路徑
            augmentor: DataAugmentor 實例（可選，訓練時使用）
        
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
            original_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frames.append(frame)

            cap.release()

            if len(original_frames) == 0:
                print(f"❌ 影片無有效幀: {input_path}")
                return False

            # 步驟1: 幀數標準化
            normalized_frames = self.normalize_frames(original_frames, self.TARGET_FRAMES)

            # 步驟2: 智能裁切人體（在 resize 之前）
            cropped_frames = []
            crop_success_count = 0
            
            for frame in normalized_frames:
                if self.enable_cropping:
                    frame_cropped, success = self.detect_and_crop_person(frame)
                    if success:
                        cropped_frames.append(frame_cropped)
                        crop_success_count += 1
                    else:
                        cropped_frames.append(frame)
                else:
                    cropped_frames.append(frame)
            
            # 如果啟用了裁切，顯示裁切成功率
            if self.enable_cropping:
                crop_rate = (crop_success_count / len(normalized_frames)) * 100
                if crop_rate < 50:
                    print(f"⚠️  警告: {os.path.basename(input_path)} 人體檢測率較低 ({crop_rate:.1f}%)")

            # 步驟3: 解析度標準化
            resized_frames = []
            for frame in cropped_frames:
                frame_resized = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
                resized_frames.append(frame_resized)

            # 步驟4: 數據增強（可選）
            if augmentor is not None:
                # 使用 DataAugmentor 創建8個增強版本
                success_count = augmentor.create_augmented_versions(
                    resized_frames,
                    output_path_base,
                    self._save_video_frames
                )
                return success_count == 8  # 所有8個版本都必須成功
            else:
                # 不使用數據增強，只輸出標準化版本
                output_file = output_path_base + self.OUTPUT_EXT
                return self._save_video_frames(resized_frames, output_file)

        except Exception as e:
            print(f"❌ 處理影片時出錯: {input_path}, 錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def process_directory(self, input_dir, output_dir, max_workers=None, augmentor=None):
        """
        處理整個目錄下的所有影片
        
        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            max_workers: 工作線程數（None=自動檢測）
            augmentor: DataAugmentor 實例（可選，訓練時使用）
        """
        if not os.path.exists(input_dir):
            print(f"❌ 輸入目錄不存在: {input_dir}")
            return
            
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # CPU 硬體優化：自動檢測 CPU 核心數
        if max_workers is None:
            cpu_count = psutil.cpu_count(logical=True)
            # 使用 CPU 核心數的 75%，避免系統過載
            max_workers = max(1, int(cpu_count * 0.75))
            print(f"🔧 檢測到 {cpu_count} 個邏輯 CPU 核心，使用 {max_workers} 個工作線程")
        
        print(f"\n📋 處理規格:")
        print(f"   - 智能裁切: {'✅ 啟用 (MediaPipe Pose)' if self.enable_cropping else '❌ 停用'}")
        if self.enable_cropping:
            print(f"      └─ 自動檢測最前面的人（上半身 + 完整手臂）")
            print(f"      └─ Padding: {self.CROP_PADDING * 100:.0f}%")
        print(f"   - 幀數: {self.TARGET_FRAMES} 幀")
        print(f"   - FPS: {self.TARGET_FPS} fps")
        print(f"   - 解析度: {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}")
        if self.has_ffmpeg:
            print(f"   - 編碼: h.264 (libx264) ✓")
        else:
            print(f"   - 編碼: mp4v (建議安裝ffmpeg以使用h264)")
        
        if augmentor is not None:
            print(f"   - 數據增強: ✅ 啟用")
            print(f"      └─ 確定性增強（原始+旋轉+縮放+亮度+水平反轉）")
            print(f"      └─ 每個樣本輸出 8 個版本")
        else:
            print(f"   - 數據增強: ❌ 停用（僅輸出標準化版本）")
        print()
        
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
                    
                    video_files.append((src_path, dest_path))
        
        print(f"📁 發現 {len(video_files)} 個影片檔案待處理\n")
        
        # 使用多線程處理影片
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(
                    lambda x: self.process_video(x[0], x[1], augmentor), 
                    video_files
                ), 
                total=len(video_files), 
                desc="🎬 處理影片"
            ))
        
        success_count = sum(results)
        print(f"\n✅ 成功處理 {success_count}/{len(video_files)} 個影片")
        
        if success_count < len(video_files):
            print(f"⚠️  失敗 {len(video_files) - success_count} 個影片")

def main():
    """
    主程式 - 訓練前處理（包含數據增強）
    
    如果只需要標準化（辨識時使用），請參考 process_for_inference()
    """
    print("=" * 70)
    print("手語影片前處理：智能裁切 + 標準化 + 數據增強（訓練用）")
    print("=" * 70)
    
    # 檢測是否在 Kaggle 環境中
    is_kaggle = os.path.exists('/kaggle')
    
    if is_kaggle:
        # Kaggle 環境配置
        input_dir = "/kaggle/input/augment/augmented_videos"
        output_dir = "/kaggle/working/videos_normalized"
        print("🌐 檢測到 Kaggle 環境")
    else:
        # 本地環境配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 使用augmented_videos作為輸入
        input_dir = os.path.join(os.path.dirname(current_dir), "/Users/baidongqu/Desktop/MVP/videos")
        output_dir = os.path.join(os.path.dirname(current_dir), "videos_normalized")
        print("💻 使用本地環境")
    
    print(f"📂 輸入目錄: {input_dir}")
    print(f"📂 輸出目錄: {output_dir}")
    print()
    
    # 初始化處理器（啟用智能裁切）
    processor = VideoProcessor(enable_cropping=True)
    
    # 初始化數據增強器（訓練時使用）
    augmentor = DataAugmentor()
    print("✅ 數據增強器已初始化")
    print()
    
    # 處理影片（包含數據增強）
    processor.process_directory(input_dir, output_dir, augmentor=augmentor)
    
    # 打包輸出目錄為zip檔案
    if is_kaggle:
        zip_path = "/kaggle/working/videos_normalized.zip"
    else:
        zip_path = os.path.join(os.path.dirname(current_dir), "videos_normalized.zip")
    
    print(f"\n📦 正在打包輸出檔案到: {zip_path}")
    
    try:
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
    print("前處理完成！")
    print("=" * 70)


def process_for_inference(input_video, output_video, enable_cropping=True):
    """
    辨識前處理 - 僅標準化（不含數據增強）
    
    用於辨識時的影片前處理：
    1. 智能裁切人體（可選）
    2. 幀數標準化（80幀）
    3. 解析度標準化（224x224）
    
    不包含數據增強！僅輸出單個標準化版本
    
    Args:
        input_video: 輸入影片路徑
        output_video: 輸出影片路徑
        enable_cropping: 是否啟用智能裁切（預設開啟）
    
    Returns:
        處理是否成功
    
    使用範例：
        >>> from processor import process_for_inference
        >>> success = process_for_inference('input.mp4', 'output.mp4')
    """
    print("=" * 70)
    print("手語影片前處理：智能裁切 + 標準化（辨識用）")
    print("=" * 70)
    
    # 初始化處理器（不使用數據增強）
    processor = VideoProcessor(enable_cropping=enable_cropping)
    
    # 處理影片（augmentor=None，只輸出標準化版本）
    success = processor.process_video(input_video, output_video, augmentor=None)
    
    if success:
        print(f"✅ 處理完成: {output_video}")
    else:
        print(f"❌ 處理失敗: {input_video}")
    
    return success


if __name__ == "__main__":
    main()
