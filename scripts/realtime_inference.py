#!/usr/bin/env python3
"""
實時攝像頭手語識別腳本

⚠️  模型版本：v3.0 - Deep Improvements（深度改進版）

🎯 v3.0 關鍵改進：
1. 多尺度池化：捕捉不同時間尺度的特徵
2. MaxNorm 約束：限制權重大小，防止過度自信
3. Focal Loss：專注於難分類樣本
4. Temperature Scaling：輸出信心度更準確（T=1.5）
5. Mixup 數據增強：訓練時樣本混合（推論時無影響）
6. Label Smoothing：防止模型過度擬合標籤

📊 預期輸出：
- 信心度範圍：70-85%（不再是 98%+）
- Top-5 分佈更均勻：反映真實不確定性
- 誤判時不會給出極端信心度

🔧 推論無需修改：
- 所有正則化技術在推論時自動處理
- BatchNorm/LayerNorm 自動切換到推論模式
- Temperature Scaling 自動應用於輸出 logits
- MaxNorm 約束不影響推論速度

功能：
1. 使用前置鏡頭錄影（80 幀，約 2.67 秒 @ 30fps）
2. 無需保存文件：直接從記憶體提取特徵
3. 並行特徵提取：RGB (MobileNetV3) + Skeleton (MediaPipe)
4. 實時推論：CPU 模式（比 MPS 快 23 倍）
5. 持續顯示最新結果（按 C 清除）
"""

import os
import sys

# ⚠️ 關鍵！在導入任何其他模組前先設置環境變數
# 禁用 MediaPipe GPU/OpenGL（容器環境無 GPU 支援）
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'  # 禁用 MediaPipe GPU
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # 替代變數
os.environ['GLOG_minloglevel'] = '2'  # 減少 Google Log 日誌
os.environ['KERAS_BACKEND'] = 'tensorflow'  # 設置 Keras backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少 TensorFlow 日誌

# 現在才導入其他模組
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import tempfile
import threading
import queue
import logging

import tensorflow as tf
import keras

# 導入模組
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "feature_extraction"))

from rgb_feature_extraction import RGBFeatureExtractor
from skeleton_feature_extraction import EnhancedSkeletonExtractor


# ==================== 自定義層定義（載入模型需要）====================
@keras.saving.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    """Focal Loss"""
    def __init__(self, num_classes=15, gamma=2.0, alpha=None, label_smoothing=0.1, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    def call(self, y_true, y_pred):
        from keras import ops
        if len(ops.shape(y_true)) == 1:
            y_true_one_hot = ops.one_hot(ops.cast(y_true, 'int32'), self.num_classes)
            if self.label_smoothing > 0:
                y_true_one_hot = y_true_one_hot * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            y_true_one_hot = y_true
        epsilon = 1e-7
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true_one_hot * ops.log(y_pred)
        p_t = ops.sum(y_true_one_hot * y_pred, axis=-1, keepdims=True)
        focal_weight = ops.power(1.0 - p_t, self.gamma)
        focal_cross_entropy = focal_weight * cross_entropy
        if self.alpha is not None:
            alpha_weight = y_true_one_hot * self.alpha
            focal_cross_entropy = alpha_weight * focal_cross_entropy
        return ops.mean(ops.sum(focal_cross_entropy, axis=-1))
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 'gamma': self.gamma, 'alpha': self.alpha, 'label_smoothing': self.label_smoothing})
        return config

@keras.saving.register_keras_serializable()
class MixupAccuracy(keras.metrics.Metric):
    """Mixup Accuracy"""
    def __init__(self, name='mixup_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        from keras import ops
        if len(ops.shape(y_true)) == 1:
            y_true_labels = ops.cast(y_true, 'int32')
        else:
            y_true_labels = ops.argmax(y_true, axis=-1)
        y_pred_labels = ops.argmax(y_pred, axis=-1)
        matches = ops.cast(ops.equal(y_true_labels, y_pred_labels), 'float32')
        self.total.assign_add(ops.sum(matches))
        self.count.assign_add(ops.cast(ops.size(matches), 'float32'))
    def result(self):
        return self.total / self.count
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

@keras.saving.register_keras_serializable()
class MixupTop3Accuracy(keras.metrics.Metric):
    """Mixup Top-3 Accuracy"""
    def __init__(self, name='mixup_top3_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        from keras import ops
        if len(ops.shape(y_true)) == 1:
            y_true_labels = ops.cast(y_true, 'int32')
        else:
            y_true_labels = ops.argmax(y_true, axis=-1)
        top3_pred = ops.top_k(y_pred, k=3)[1]
        y_true_expanded = ops.expand_dims(y_true_labels, axis=-1)
        matches = ops.any(ops.equal(top3_pred, y_true_expanded), axis=-1)
        matches = ops.cast(matches, 'float32')
        self.total.assign_add(ops.sum(matches))
        self.count.assign_add(ops.cast(ops.size(matches), 'float32'))
    def result(self):
        return self.total / self.count
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class RealtimeSignLanguageRecognition:
    """實時手語識別器"""
    
    # 錄影參數（與 VideoProcessor 一致）
    TARGET_FRAMES = 80
    TARGET_FPS = 30
    TARGET_WIDTH = 224
    TARGET_HEIGHT = 224

    def _setup_logging(self, log_file=None, enable_logging=True):
        """設置日誌系統"""
        if not enable_logging:
            # 如果禁用日誌，只設置基本的控制台輸出
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
            self.logger = logging.getLogger(__name__)
            self.log_file = None
            return
            
        # 創建 logs 目錄
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # 如果沒有指定 log_file，使用默認路徑
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = logs_dir / f"realtime_inference_{timestamp}.log"

        # 配置根日誌器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)  # 同時輸出到控制台
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.log_file = log_file

        self.logger.info(f"📝 日誌文件: {log_file}")
        self.logger.info(f"🚀 程序啟動時間: {datetime.now()}")
        self.logger.info(f"🔧 系統信息: Python {sys.version.split()[0]} on {sys.platform}")

    def __init__(self, model_path, label_map_path, device='mps', camera_id=0, log_file=None, enable_logging=False):
        """
        初始化實時識別器

        Args:
            model_path: 模型路徑
            label_map_path: 標籤映射路徑
            device: 設備類型 ('mps', 'gpu', 'cpu')
            camera_id: 攝像頭 ID（0=前置鏡頭，1=後置鏡頭）
            log_file: 日誌文件路徑（如果為 None，則使用默認路徑）
            enable_logging: 是否啟用日誌記錄（預設為 False）
        """
        self.device = device
        self.camera_id = camera_id

        # 設置日誌
        self._setup_logging(log_file, enable_logging)

        self.logger.info("=" * 70)
        self.logger.info("🎥 實時手語識別系統")
        self.logger.info("=" * 70)
        
        # 載入模型
        self._load_model(model_path, label_map_path)
        
        # 初始化處理器
        self._init_processors()
        
        # 錄影狀態
        self.is_recording = False
        self.frame_count = 0
        
        # 使用預分配的循環緩衝區（避免動態 list 增長）
        self.frame_buffer = np.zeros(
            (self.TARGET_FRAMES, self.TARGET_HEIGHT, self.TARGET_WIDTH, 3), 
            dtype=np.uint8
        )
        
        # 結果隊列（用於線程間通信）
        self.result_queue = queue.Queue()
        
        # 最新的識別結果（持久保存）
        self.latest_result = None
        
        # 模型預熱
        self._warmup_model()
        
        self.logger.info(f"✅ 系統初始化完成！")
        self.logger.info(f"📹 攝像頭 ID: {camera_id}")
        self.logger.info(f"🎬 錄影規格: {self.TARGET_FRAMES} 幀 @ {self.TARGET_FPS} fps, {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}")
        self.logger.info("=" * 70)
    
    def _load_model(self, model_path, label_map_path):
        """載入模型和標籤"""
        import json

        self.logger.info(f"📥 載入模型: {model_path}")

        # 設置 TensorFlow 設備
        if self.device == 'mps':
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                self.logger.info(f"🚀 使用 M1 MPS 加速")
            else:
                self.logger.warning("⚠️  MPS 不可用，使用 CPU")
                self.device = 'cpu'

        # 啟用 unsafe deserialization（處理 Lambda 層）
        keras.config.enable_unsafe_deserialization()
        
        # 載入模型
        custom_objects = {
            'FocalLoss': FocalLoss,
            'MixupAccuracy': MixupAccuracy,
            'MixupTop3Accuracy': MixupTop3Accuracy
        }
        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        keras.mixed_precision.set_global_policy('float32')

        # 載入標籤
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        self.idx_to_word = {v: k for k, v in self.label_map.items()}

        self.logger.info(f"✅ 模型載入成功（{len(self.label_map)} 個單詞）")
    
    def _warmup_model(self):
        """
        預熱模型（首次推論編譯優化）
        
        注意：BiGRU 模型在 TensorFlow Metal (MPS) 上推論極慢（~27s）
        CPU 反而快 23 倍（~1.1s），因此強制使用 CPU 推論
        """
        self.logger.info("🔥 預熱模型...")
        self.logger.info("⚠️  注意：BiGRU 在 MPS 上很慢，使用 CPU 推論（快 23 倍）")
        
        dummy_input = np.zeros((1, 300, 1119), dtype=np.float32)
        
        # 強制使用 CPU 推論（比 MPS 快）
        with tf.device('/CPU:0'):
            _ = self.model.predict(dummy_input, verbose=0)
        
        self.logger.info("✅ 模型預熱完成")
    
    def _init_processors(self):
        """初始化處理器"""
        self.logger.info("\n🔧 初始化處理器...")

        # RGB 特徵提取器（PyTorch - MPS）
        if self.device == 'mps':
            torch_device = torch.device('mps')
            device_type = 'gpu'
        elif self.device == 'gpu':
            torch_device = torch.device('cuda')
            device_type = 'gpu'
        else:
            torch_device = torch.device('cpu')
            device_type = 'cpu'

        self.rgb_extractor = RGBFeatureExtractor(torch_device, device_type)

        # 骨架特徵提取器（MediaPipe Holistic - 159維，自動使用 Metal 加速）
        from skeleton_feature_extraction import EnhancedSkeletonExtractor
        self.skeleton_extractor = EnhancedSkeletonExtractor(num_threads=4)

        self.logger.info("✅ 處理器初始化完成")
    
    def start_camera(self):
        """啟動攝像頭"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 無法打開攝像頭 {self.camera_id}")
        
        # 設置攝像頭參數
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.logger.info(f"📹 攝像頭已啟動: {actual_width}x{actual_height} @ {actual_fps} fps")
        
        return True
    
    def stop_camera(self):
        """關閉攝像頭"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def start_recording(self):
        """開始錄影"""
        self.is_recording = True
        self.frame_count = 0
        # 重置緩衝區（不需要，因為會被覆蓋）
        self.logger.info(f"\n🔴 開始錄影（目標 {self.TARGET_FRAMES} 幀）...")
    
    def stop_recording(self):
        """停止錄影"""
        self.is_recording = False
        self.logger.info(f"⏹️  停止錄影（已錄 {self.frame_count} 幀）")
    
    def add_frame(self, frame):
        """添加幀到錄影緩衝區（優化版：錄影時同步 resize）"""
        if self.is_recording and self.frame_count < self.TARGET_FRAMES:
            # 直接 resize 到目標尺寸並寫入緩衝區
            frame_resized = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
            
            # 轉換為 RGB（MediaPipe 和模型都需要 RGB）
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # 寫入預分配的緩衝區
            self.frame_buffer[self.frame_count] = frame_rgb
            self.frame_count += 1
            
            # 達到目標幀數後自動停止並開始處理
            if self.frame_count >= self.TARGET_FRAMES:
                self.stop_recording()
                # 在新線程中處理視頻（避免阻塞主線程）
                threading.Thread(target=self._process_video_optimized, daemon=True).start()
    
    def _process_video_optimized(self):
        """
        優化版：處理錄製的視頻（在後台線程運行）
        ✅ 直接從記憶體中的 frames 提取特徵
        ✅ 無需保存/讀取臨時文件
        ✅ 減少 I/O 開銷 ~1000ms
        """
        import time
        start_time = time.time()
        self.logger.info(f"\n⚙️  開始處理視頻（優化模式）...")

        try:
            # 1. 準備 frames（已經是 224x224 RGB）
            # 取出實際錄製的幀數
            frames = [self.frame_buffer[i] for i in range(self.frame_count)]
            self.logger.info(f"📊 處理 {len(frames)} 幀 ({self.TARGET_WIDTH}x{self.TARGET_HEIGHT} RGB)")
            
            # 2. 並行特徵提取（直接從 frames）
            self.logger.info("🔄 並行特徵提取（無 I/O）...")
            
            rgb_features = None
            skeleton_features = None
            errors = []
            
            def extract_rgb():
                nonlocal rgb_features, errors
                try:
                    t0 = time.time()
                    self.logger.info("  📸 提取 RGB 特徵...")
                    rgb_features = self.rgb_extractor.extract_features_from_frames(frames)
                    t1 = time.time()
                    self.logger.info(f"  ✅ RGB 特徵: {rgb_features.shape} ({(t1-t0)*1000:.0f}ms)")
                except Exception as e:
                    errors.append(f"RGB 提取失敗: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            def extract_skeleton():
                nonlocal skeleton_features, errors
                try:
                    t0 = time.time()
                    self.logger.info("  🦴 提取骨架特徵...")
                    # 傳入原始幀尺寸用於正規化
                    skeleton_features = self.skeleton_extractor.extract_features_from_frames(
                        frames, 
                        frame_width=self.TARGET_WIDTH, 
                        frame_height=self.TARGET_HEIGHT
                    )
                    t1 = time.time()
                    self.logger.info(f"  ✅ 骨架特徵: {skeleton_features.shape} ({(t1-t0)*1000:.0f}ms)")
                except Exception as e:
                    errors.append(f"骨架提取失敗: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # 並行執行
            rgb_thread = threading.Thread(target=extract_rgb)
            skeleton_thread = threading.Thread(target=extract_skeleton)
            
            rgb_thread.start()
            skeleton_thread.start()
            
            rgb_thread.join()
            skeleton_thread.join()
            
            # 檢查錯誤
            if errors:
                error_msg = "; ".join(errors)
                raise RuntimeError(error_msg)
            
            if rgb_features is None or skeleton_features is None:
                raise ValueError("特徵提取失敗")
            
            # 3. 融合特徵
            self.logger.info("🔀 融合特徵...")
            min_len = min(len(rgb_features), len(skeleton_features))
            rgb_features = rgb_features[:min_len]
            skeleton_features = skeleton_features[:min_len]
            
            concat_features = np.concatenate([rgb_features, skeleton_features], axis=1)
            
            # Padding 到 300（模型輸入要求）
            max_length = 300
            if len(concat_features) < max_length:
                padding = np.zeros((max_length - len(concat_features), 1119), dtype=np.float32)
                concat_features = np.concatenate([concat_features, padding], axis=0)
            else:
                concat_features = concat_features[:max_length]
            
            self.logger.info(f"✅ 特徵融合完成: {concat_features.shape}")
            
            # 4. 推論（強制使用 CPU，比 MPS 快 23 倍）
            t0 = time.time()
            self.logger.info("🤖 執行推論（CPU 模式）...")
            features_batch = np.expand_dims(concat_features, axis=0)
            
            # 強制使用 CPU（BiGRU 在 CPU 上比 MPS 快 23 倍）
            with tf.device('/CPU:0'):
                predictions = self.model.predict(features_batch, verbose=0)
            
            t1 = time.time()
            self.logger.info(f"✅ 推論完成 ({(t1-t0)*1000:.0f}ms)")
            
            # 5. 獲取 Top-5 結果
            top_indices = np.argsort(predictions[0])[::-1][:5]
            results = [(self.idx_to_word[idx], float(predictions[0][idx])) for idx in top_indices]
            
            # 6. 將結果放入隊列
            self.result_queue.put({
                'success': True,
                'results': results,
                'timestamp': datetime.now()
            })
            
            # 7. 在終端打印結果
            total_time = time.time() - start_time
            result_text = "\n" + "=" * 70 + "\n🎯 辨識結果:\n" + "=" * 70
            for i, (word, confidence) in enumerate(results, 1):
                bar = "█" * int(confidence * 50)
                result_text += f"\n{i}. {word:15s} {confidence*100:6.2f}% {bar}"
            result_text += f"\n" + "=" * 70
            result_text += f"\n⏱️  總耗時: {total_time*1000:.0f}ms"
            result_text += f"\n✅ 推論完成！按 SPACE 繼續錄影..."
            result_text += "\n" + "=" * 70
            
            self.logger.info(result_text)
        
        except Exception as e:
            self.logger.error(f"❌ 處理失敗: {e}")
            import traceback
            self.logger.error(f"詳細錯誤信息:\n{traceback.format_exc()}")
            self.result_queue.put({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
    
    
    def draw_ui(self, frame):
        """繪製 UI（顯示狀態和結果）"""
        # 創建半透明的狀態欄
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # 頂部狀態欄
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 顯示狀態
        if self.is_recording:
            status_text = f"Recording... ({self.frame_count}/{self.TARGET_FRAMES})"
            color = (0, 0, 255)  # 紅色
            
            # 進度條
            progress = self.frame_count / self.TARGET_FRAMES
            bar_width = int(width * 0.8)
            bar_x = int(width * 0.1)
            cv2.rectangle(frame, (bar_x, 50), (bar_x + bar_width, 70), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, 50), (bar_x + int(bar_width * progress), 70), color, -1)
        else:
            status_text = "Ready - Press SPACE to start"
            color = (0, 255, 0)  # 綠色
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 從隊列中獲取最新結果（如果有）
        try:
            while not self.result_queue.empty():
                self.latest_result = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        
        # 顯示最新的識別結果（持續顯示）
        if self.latest_result is not None:
            if self.latest_result['success']:
                # 顯示 Top-3 結果
                y_offset = height - 150
                cv2.rectangle(overlay, (0, y_offset), (width, height), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                cv2.putText(frame, "Results:", (10, y_offset + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                for i, (word, conf) in enumerate(self.latest_result['results'][:3]):
                    text = f"{i+1}. {word}: {conf*100:.1f}%"
                    y = y_offset + 60 + i * 30
                    cv2.putText(frame, text, (10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # 顯示錯誤
                cv2.putText(frame, f"Error: {self.latest_result['error']}", (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 說明文字
        help_text = [
            "SPACE: Record",
            "C: Clear",
            "Q: Quit"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (width - 250, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """主循環"""
        try:
            # 啟動攝像頭
            self.start_camera()

            self.logger.info("\n" + "=" * 70)
            self.logger.info("📹 攝像頭已啟動！")
            self.logger.info("=" * 70)
            self.logger.info("使用說明：")
            self.logger.info("  - 按 SPACE 開始錄影（自動錄製 80 幀）")
            self.logger.info("  - 錄影完成後自動處理並顯示結果")
            self.logger.info("  - 按 C 清除辨識結果")
            self.logger.info("  - 按 Q 退出")
            self.logger.info("=" * 70)

            while True:
                # 讀取幀
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("❌ 讀取幀失敗")
                    break
                
                # 水平翻轉（前置鏡頭鏡像效果）
                frame = cv2.flip(frame, 1)
                
                # 添加幀到錄影緩衝區（如果正在錄影）
                if self.is_recording:
                    self.add_frame(frame)
                
                # 繪製 UI
                display_frame = self.draw_ui(frame)
                
                # 顯示
                cv2.imshow('Sign Language Recognition - Realtime', display_frame)
                
                # 按鍵處理
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    # 退出
                    break
                elif key == ord(' '):
                    # 空格鍵：開始/停止錄影
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                elif key == ord('c') or key == ord('C'):
                    # C 鍵：清除結果
                    self.latest_result = None
                    self.logger.info("\n🗑️  已清除辨識結果")

        finally:
            # 清理
            self.stop_camera()
            cv2.destroyAllWindows()
            self.logger.info("\n👋 程序已退出")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='實時攝像頭手語識別')
    parser.add_argument('--model', type=str, 
                       default='model_output/best_model_mps.keras',
                       help='模型路徑')
    parser.add_argument('--labels', type=str,
                       default='model_output/label_map.json',
                       help='標籤映射路徑')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['mps', 'gpu', 'cpu'],
                       help='設備類型')
    parser.add_argument('--camera', type=int, default=0,
                       help='攝像頭 ID（0=前置，1=後置）')
    parser.add_argument('--log-file', type=str, default=None,
                       help='日誌文件路徑（默認使用 logs/realtime_inference_YYYYMMDD_HHMMSS.log）')
    parser.add_argument('--enable-log', action='store_true',
                       help='啟用日誌記錄（預設禁用）')
    
    args = parser.parse_args()
    
    # 檢查文件
    model_path = Path(args.model)
    label_path = Path(args.labels)
    
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        print("請先執行: python3 convert_model_for_mps.py")
        return

    if not label_path.exists():
        print(f"❌ 標籤映射不存在: {label_path}")
        return
    
    # 創建識別器並運行
    recognizer = RealtimeSignLanguageRecognition(
        model_path=model_path,
        label_map_path=label_path,
        device=args.device,
        camera_id=args.camera,
        log_file=args.log_file,
        enable_logging=args.enable_log  # 只有在指定 --enable-log 時才啟用
    )
    
    recognizer.run()


if __name__ == "__main__":
    main()

