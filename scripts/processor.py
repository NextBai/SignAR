"""
手語影片前處理腳本
功能：
1. 影片標準化：
   - 幀數：統一為80幀（使用智能採樣/插值）
   - FPS：統一為30fps
   - 解析度：統一為224x224
   - 編碼：h.264 (libx264)
2. 極輕微數據增強：
   - 極輕微旋轉 (±2°，20%機率)
   - 極輕微縮放 (0.98-1.02x，10%機率)
   - 極輕微亮度調整 (±5%，15%機率)
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import zipfile
import psutil

class VideoProcessor:
    # 標準化參數
    TARGET_FRAMES = 80      # 目標幀數
    TARGET_FPS = 30         # 目標幀率
    TARGET_WIDTH = 224      # 目標寬度
    TARGET_HEIGHT = 224     # 目標高度
    TARGET_BITRATE = 2000   # 目標比特率 (kbps)
    OUTPUT_EXT = '.mp4'     # 統一輸出格式
    
    def __init__(self):
        # 檢查是否有ffmpeg（用於後處理確保h264編碼）
        self.has_ffmpeg = self._check_ffmpeg()
        
        # 數據增強參數 - 極為保守
        self.augmentation_params = {
            'rotation_range': (-2, 2),      # 極輕微旋轉角度範圍（度）
            'scale_range': (0.98, 1.02),    # 極輕微縮放範圍
            'brightness_range': (0.95, 1.05), # 極輕微亮度調整範圍
        }
        
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

    def apply_augmentation(self, frame, augmentation_type=None):
        """
        確定性數據增強 - 為每個樣本創建多個增強版本

        Args:
            frame: 輸入幀 [H, W, 3]
            augmentation_type: 增強類型 ('rotation', 'scale', 'brightness', 'flip', None)

        Returns:
            增強後的幀 [H, W, 3]
        """
        augmented = frame.astype(np.float32)

        if augmentation_type == 'rotation':
            # 確定性旋轉 (±2°)
            angle = np.random.RandomState(42).choice([-2, -1, 1, 2])  # 固定種子確保可重現
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
            augmented = cv2.flip(augmented, 1)  # 1表示水平反轉

        # 如果沒有指定增強類型，返回原始幀
        return augmented.astype(np.uint8)

    def process_frames(self, frames):
        """
        處理幀序列：標準化 + 確定性數據增強

        為每個輸入幀創建4個版本：
        1. 原始版本
        2. 旋轉增強版本
        3. 縮放增強版本
        4. 亮度增強版本

        Args:
            frames: 輸入幀列表

        Returns:
            處理後的幀列表（4倍數量）
        """
        processed_frames = []

        for frame in frames:
            # 1. 解析度標準化
            frame_resized = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))

            # 2. 創建4個增強版本
            # 原始版本
            processed_frames.append(frame_resized)

            # 旋轉增強版本
            frame_rotation = self.apply_augmentation(frame_resized, 'rotation')
            processed_frames.append(frame_rotation)

            # 縮放增強版本
            frame_scale = self.apply_augmentation(frame_resized, 'scale')
            processed_frames.append(frame_scale)

            # 亮度增強版本
            frame_brightness = self.apply_augmentation(frame_resized, 'brightness')
            processed_frames.append(frame_brightness)

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

    def process_video(self, input_path, output_path):
        """
        處理單一影片：標準化 + 確定性增強

        為每個輸入影片創建8個增強版本：
        1. 原始版本: video.mp4
        2. 旋轉增強版本: video_rotation.mp4
        3. 縮放增強版本: video_scale.mp4
        4. 亮度增強版本: video_brightness.mp4
        5. 原始+反轉版本: video_flip.mp4
        6. 旋轉+反轉版本: video_rotation_flip.mp4
        7. 縮放+反轉版本: video_scale_flip.mp4
        8. 亮度+反轉版本: video_brightness_flip.mp4

        標準化規格：
        - 80幀
        - 30fps
        - 224x224
        - h.264編碼
        - 2000kbps比特率
        - 統一.mp4格式

        Args:
            input_path: 輸入影片路徑（支援.mp4, .avi, .mov, .wmv等）
            output_path: 輸出影片路徑（將作為基礎路徑創建多個文件）

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

            # 步驟2: 解析度標準化
            resized_frames = []
            for frame in normalized_frames:
                frame_resized = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
                resized_frames.append(frame_resized)

            # 步驟3: 創建8個增強版本並輸出
            base_augmentation_types = [None, 'rotation', 'scale', 'brightness']
            base_suffixes = ['', '_rotation', '_scale', '_brightness']

            success_count = 0

            for base_aug, base_suffix in zip(base_augmentation_types, base_suffixes):
                # 創建基本增強版本
                if base_aug is None:
                    base_frames = resized_frames
                else:
                    base_frames = [self.apply_augmentation(frame, base_aug) for frame in resized_frames]

                # 創建不反轉版本
                output_file = output_path_base + base_suffix + self.OUTPUT_EXT
                if self._save_video_frames(base_frames, output_file):
                    success_count += 1

                # 創建反轉版本
                flip_frames = [self.apply_augmentation(frame, 'flip') for frame in base_frames]
                flip_suffix = base_suffix + '_flip' if base_suffix else '_flip'
                output_file_flip = output_path_base + flip_suffix + self.OUTPUT_EXT
                if self._save_video_frames(flip_frames, output_file_flip):
                    success_count += 1

            return success_count == 8  # 所有8個版本都必須成功

        except Exception as e:
            print(f"❌ 處理影片時出錯: {input_path}, 錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def process_directory(self, input_dir, output_dir, max_workers=None):
        """
        處理整個目錄下的所有影片
        
        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            max_workers: 工作線程數（None=自動檢測）
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
        
        print(f"\n📋 標準化規格:")
        print(f"   - 幀數: {self.TARGET_FRAMES} 幀")
        print(f"   - FPS: {self.TARGET_FPS} fps")
        print(f"   - 解析度: {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}")
        print(f"   - 編碼: {self.OUTPUT_EXT}")
        if self.has_ffmpeg:
            print(f"   - 編碼: h.264 (libx264) ✓")
        else:
            print(f"   - 編碼: mp4v (建議安裝ffmpeg以使用h264)")
        print(f"   - 數據增強: 確定性增強 (原始+旋轉+縮放+亮度+水平反轉，每個樣本8個版本)")
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
                    lambda x: self.process_video(x[0], x[1]), 
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
    """主程式"""
    print("=" * 70)
    print("手語影片前處理：標準化 + 微量數據增強")
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
    
    # 初始化處理器並處理影片
    processor = VideoProcessor()
    processor.process_directory(input_dir, output_dir)
    
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

if __name__ == "__main__":
    main()
