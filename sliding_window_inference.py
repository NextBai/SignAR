#!/usr/bin/env python3
"""
滑動窗口手語識別腳本

⚠️  模型版本：v3.0 - Deep Improvements（深度改進版）

🎯 v3.0 關鍵改進：
1. 多尺度池化：GlobalMaxPool + GlobalAvgPool 並聯
2. MaxNorm 約束：防止極端 logits（Embedding=3.0, Logits=1.5）
3. Focal Loss (γ=2.0)：降低簡單樣本權重
4. Temperature Scaling (T=1.5)：校準信心度分佈
5. Mixup (α=0.2)：batch 級別樣本混合（訓練時）
6. Label Smoothing (ε=0.1)：軟化 one-hot 標籤

📊 預期改善：
- 信心度：70-85%（原 98%+ 過於自信）
- 準確率：92-96%（原 100% 表示過擬合）
- Top-1 到 Top-5 概率更平滑
- 相似手勢（如 teacher/student）不再極端誤判

🔧 推論特性：
- 所有改進在推論時自動生效
- BatchNorm/LayerNorm 自動切換推論模式
- Temperature Scaling 層透明處理
- 無需代碼修改

功能：
1. 讀取任意長度影片（最短 80 幀）
2. 滑動窗口掃描：80 幀/窗口，可調步長
3. 每個窗口獨立推論：Top-5 結果
4. 時間軸可視化：完整輸出所有窗口結果
5. JSON 結果保存：便於後續分析
"""

import os

# 🚫 禁用 GPU/Metal/OpenGL - 必須在所有 import 之前設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['MEDIAPIPE_DISABLE_EGL'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['GLOG_logtostderr'] = '1'
# 抑制 MediaPipe GPU 試探的錯誤訊息（2=只顯示 ERROR 以上）
os.environ['GLOG_minloglevel'] = '2'

# 設置 Keras backend 為 TensorFlow
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import json
import time
from openai import OpenAI

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


class SlidingWindowInference:
    """滑動窗口手語識別器"""
    
    # 參數配置
    WINDOW_SIZE = 80        # 每個窗口 80 幀（約 2.67 秒 @ 30fps）
    TARGET_FPS = 30         # 目標幀率
    TARGET_WIDTH = 224      # 目標寬度
    TARGET_HEIGHT = 224     # 目標高度
    
    def __init__(self, model_path, label_map_path, device='mps', stride=60, openai_api_key=None, progress_callback=None):
        """
        初始化滑動窗口識別器
        
        Args:
            model_path: 模型路徑
            label_map_path: 標籤映射路徑
            device: 設備類型（用於特徵提取器，推論強制使用 CPU）
            stride: 滑動步長（幀數）
                   - 80 幀：無重疊，最快，適合快速掃描
                   - 60 幀（預設）：25% 重疊，平衡精度和速度
                   - 40 幀：50% 重疊，更密集檢測但計算量大
            openai_api_key: OpenAI API 金鑰（用於句子重組）
            progress_callback: 進度回調函數，參數為 (current, total, message)
        
        注意：訓練數據平均單詞長度為 88 幀（3.1 秒）
        """
        self.device = device
        self.stride = stride
        self.openai_client = None
        self.progress_callback = progress_callback
        
        # 初始化 OpenAI
        if openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                print("✅ OpenAI 客戶端初始化成功")
            except Exception as e:
                print(f"⚠️ OpenAI 客戶端初始化失敗: {e}")
                self.openai_client = None
        
        print("=" * 70)
        print("🎬 滑動窗口手語識別系統 - 方案 C（單尺度優化）")
        print("=" * 70)
        overlap_percent = (self.WINDOW_SIZE - self.stride) / self.WINDOW_SIZE * 100
        print(f"窗口大小: {self.WINDOW_SIZE} 幀 (~{self.WINDOW_SIZE/self.TARGET_FPS:.2f} 秒)")
        print(f"滑動步長: {self.stride} 幀 (~{self.stride/self.TARGET_FPS:.2f} 秒)")
        print(f"重疊程度: {overlap_percent:.0f}% - 平衡精度與速度")
        print(f"支援任意長度影片（最短需 {self.WINDOW_SIZE} 幀）")
        print("=" * 70)
        
        # 載入模型
        self._load_model(model_path, label_map_path)
        
        # 延遲初始化特徵提取器（首次使用時才載入）
        self.rgb_extractor = None
        self.skeleton_extractor = None
        
        # 延遲預熱模型（首次推論時自動預熱）
        self._model_warmed_up = False
        
        print("✅ 系統初始化完成！（特徵提取器將延遲載入）\n")
    
    def _load_model(self, model_path, label_map_path):
        """載入模型和標籤"""
        print(f"📥 載入模型: {model_path}")

        # 啟用 unsafe deserialization（處理 Lambda 層）
        keras.config.enable_unsafe_deserialization()

        # 載入模型
        custom_objects = {
            'FocalLoss': FocalLoss,
            'MixupAccuracy': MixupAccuracy,
            'MixupTop3Accuracy': MixupTop3Accuracy
        }

        # 抑制 Keras 警告（masking 和 optimizer 變數不匹配）
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='keras')

        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        keras.mixed_precision.set_global_policy('float32')

        # 恢復警告（只抑制載入時的警告）
        warnings.filterwarnings('default', category=UserWarning, module='keras')
        
        # 載入標籤
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        self.idx_to_word = {v: k for k, v in self.label_map.items()}
        
        print(f"✅ 模型載入成功（{len(self.label_map)} 個單詞）")
    
    def _ensure_extractors_initialized(self):
        """確保特徵提取器已初始化（延遲載入）"""
        if self.rgb_extractor is None or self.skeleton_extractor is None:
            print("🔧 初始化特徵提取器...")
            
            # RGB 特徵提取器
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
            
            # 骨架特徵提取器
            self.skeleton_extractor = EnhancedSkeletonExtractor(num_threads=4)
            
            print("✅ 特徵提取器初始化完成")
    
    def _ensure_model_warmed_up(self):
        """確保模型已預熱（延遲預熱）"""
        if not self._model_warmed_up:
            print("🔥 預熱模型（CPU 模式）...")
            dummy_input = np.zeros((1, 300, 1119), dtype=np.float32)
            
            with tf.device('/CPU:0'):
                _ = self.model.predict(dummy_input, verbose=0)
            
            self._model_warmed_up = True
            print("✅ 模型預熱完成")
    
    def load_and_normalize_video(self, video_path):
        """
        讀取並標準化影片
        
        優化策略：線性插值至 80 幀的倍數
        - 原因：確保所有窗口都是完整的 80 幀
        - 範例：
          • 75 幀 → 80 幀（1個窗口）
          • 150 幀 → 160 幀（2個窗口）
          • 200 幀 → 240 幀（3個窗口）

        Args:
            video_path: 影片路徑

        Returns:
            frames: 標準化後的幀列表 (T, 224, 224, 3) RGB
        """
        print(f"\n📹 讀取影片: {Path(video_path).name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片: {video_path}")

        # 獲取影片資訊
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        print(f"  原始規格: {total_frames} 幀, {original_fps:.2f} fps, {duration:.2f} 秒")

        # 讀取所有幀
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)

        cap.release()

        if len(all_frames) == 0:
            raise RuntimeError(f"影片無有效幀: {video_path}")

        # 計算目標幀數：線性插值至 80 幀的倍數
        original_count = len(all_frames)
        
        # 計算需要多少個窗口（無條件進位）
        num_windows = int(np.ceil(original_count / self.WINDOW_SIZE))
        
        # 目標幀數 = 窗口數量 × 80
        target_frame_count = num_windows * self.WINDOW_SIZE
        
        print(f"  🎯 線性插值策略:")
        print(f"     原始幀數: {original_count}")
        print(f"     窗口數量: {num_windows} 個（每個 {self.WINDOW_SIZE} 幀）")
        print(f"     目標幀數: {target_frame_count} 幀")
        
        # 線性插值重採樣
        indices = np.linspace(0, original_count - 1, target_frame_count)
        
        resampled_frames = []
        for idx in indices:
            # 如果是整數索引，直接取幀
            if idx == int(idx):
                resampled_frames.append(all_frames[int(idx)])
            else:
                # 線性插值
                lower_idx = int(np.floor(idx))
                upper_idx = int(np.ceil(idx))
                weight = idx - lower_idx
                
                # 混合兩幀
                lower_frame = all_frames[lower_idx].astype(np.float32)
                upper_frame = all_frames[upper_idx].astype(np.float32)
                interpolated = (1 - weight) * lower_frame + weight * upper_frame
                resampled_frames.append(interpolated.astype(np.uint8))
        
        # Resize 並轉換為 RGB
        normalized_frames = []
        for frame in resampled_frames:
            frame_resized = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            normalized_frames.append(frame_rgb)

        # 計算最終的等效FPS
        final_duration = len(normalized_frames) / self.TARGET_FPS
        effective_fps = len(normalized_frames) / final_duration if final_duration > 0 else original_fps

        print(f"  ✅ 處理完成: {len(normalized_frames)} 幀 @ {effective_fps:.2f} fps")
        print(f"  📈 幀數變化: {total_frames} → {len(normalized_frames)} ({len(normalized_frames) - total_frames:+d})")

        return normalized_frames
    
    def extract_window_features(self, frames):
        """
        提取窗口特徵（並行 RGB + Skeleton）

        Args:
            frames: 窗口幀列表 (80, 224, 224, 3)

        Returns:
            features: (300, 1119) 特徵矩陣
        """
        # 確保特徵提取器已初始化
        self._ensure_extractors_initialized()

        # 並行提取特徵
        rgb_features = None
        skeleton_features = None
        errors = []
        
        def extract_rgb():
            nonlocal rgb_features, errors
            try:
                rgb_features = self.rgb_extractor.extract_features_from_frames(frames)
            except Exception as e:
                errors.append(f"RGB: {e}")
        
        def extract_skeleton():
            nonlocal skeleton_features, errors
            try:
                skeleton_features = self.skeleton_extractor.extract_features_from_frames(
                    frames, 
                    frame_width=self.TARGET_WIDTH,
                    frame_height=self.TARGET_HEIGHT
                )
            except Exception as e:
                errors.append(f"Skeleton: {e}")
        
        import threading
        rgb_thread = threading.Thread(target=extract_rgb)
        skeleton_thread = threading.Thread(target=extract_skeleton)
        
        rgb_thread.start()
        skeleton_thread.start()
        
        rgb_thread.join()
        skeleton_thread.join()
        
        if errors:
            raise RuntimeError("; ".join(errors))
        
        if rgb_features is None or skeleton_features is None:
            raise ValueError("特徵提取失敗")
        
        # 融合特徵
        min_len = min(len(rgb_features), len(skeleton_features))
        rgb_features = rgb_features[:min_len]
        skeleton_features = skeleton_features[:min_len]
        
        concat_features = np.concatenate([rgb_features, skeleton_features], axis=1)
        
        # Padding 到 300
        max_length = 300
        if len(concat_features) < max_length:
            padding = np.zeros((max_length - len(concat_features), 1119), dtype=np.float32)
            concat_features = np.concatenate([concat_features, padding], axis=0)
        else:
            concat_features = concat_features[:max_length]
        
        return concat_features
    
    def predict_window(self, features):
        """
        對單個窗口進行推論

        Args:
            features: (300, 1119) 特徵矩陣

        Returns:
            top5: [(單詞, 信心度), ...] Top-5 結果
        """
        # 確保模型已預熱
        self._ensure_model_warmed_up()

        features_batch = np.expand_dims(features, axis=0)

        # 強制使用 CPU 推論（BiGRU 在 CPU 上比 MPS 快 23 倍）
        with tf.device('/CPU:0'):
            predictions = self.model.predict(features_batch, verbose=0)

        # 獲取 Top-5
        top_indices = np.argsort(predictions[0])[::-1][:5]
        results = [(self.idx_to_word[idx], float(predictions[0][idx])) for idx in top_indices]

        return results
    
    def process_video(self, video_path, save_results=True):
        """
        處理整個影片（滑動窗口）
        
        Args:
            video_path: 影片路徑
            save_results: 是否保存結果到 JSON
        
        Returns:
            results: 所有窗口的辨識結果
        """
        start_time = time.time()
        
        # 1. 讀取並標準化影片
        frames = self.load_and_normalize_video(video_path)
        total_frames = len(frames)
        
        # 2. 計算窗口數量
        num_windows = (total_frames - self.WINDOW_SIZE) // self.stride + 1
        if num_windows <= 0:
            raise ValueError(f"影片太短，無法創建窗口（需要至少 {self.WINDOW_SIZE} 幀）")
        
        print(f"\n🔄 開始滑動窗口推論...")
        print(f"  總幀數: {total_frames}")
        print(f"  窗口數量: {num_windows}")
        print(f"  每個窗口: {self.WINDOW_SIZE} 幀 ({self.WINDOW_SIZE / self.TARGET_FPS:.2f} 秒)")
        print("=" * 70)
        
        # 發送初始進度
        if self.progress_callback:
            self.progress_callback(0, num_windows, "開始處理影片")
        
        # 3. 遍歷所有窗口
        all_results = []
        
        for i in range(num_windows):
            window_start = i * self.stride
            window_end = window_start + self.WINDOW_SIZE
            
            # 計算時間範圍
            time_start = window_start / self.TARGET_FPS
            time_end = window_end / self.TARGET_FPS
            
            print(f"\n窗口 {i+1}/{num_windows} - 時間: {time_start:.2f}s - {time_end:.2f}s")
            
            # 發送窗口開始進度
            if self.progress_callback:
                self.progress_callback(i, num_windows, f"處理窗口 {i+1}/{num_windows}")
            
            # 提取窗口幀
            window_frames = frames[window_start:window_end]
            
            # 提取特徵
            t0 = time.time()
            try:
                features = self.extract_window_features(window_frames)
                t1 = time.time()
                print(f"  ✅ 特徵提取: {(t1-t0)*1000:.0f}ms")
                
                # 發送特徵提取完成進度
                if self.progress_callback:
                    self.progress_callback(i + 0.3, num_windows, f"特徵提取完成 - 窗口 {i+1}")
                
                # 推論
                t0 = time.time()
                top5 = self.predict_window(features)
                t1 = time.time()
                print(f"  ✅ 推論: {(t1-t0)*1000:.0f}ms")
                
                # 發送推論完成進度
                if self.progress_callback:
                    self.progress_callback(i + 0.7, num_windows, f"推論完成 - 窗口 {i+1}")
                
                # 顯示結果
                print(f"  🎯 Top-5 結果:")
                for j, (word, conf) in enumerate(top5, 1):
                    bar = "█" * int(conf * 30)
                    print(f"     {j}. {word:12s} {conf*100:5.2f}% {bar}")
                
                # 保存結果
                window_result = {
                    'window_id': i,
                    'frame_start': window_start,
                    'frame_end': window_end,
                    'time_start': round(time_start, 2),
                    'time_end': round(time_end, 2),
                    'top5': [{'word': w, 'confidence': round(c, 4)} for w, c in top5]
                }
                all_results.append(window_result)
                
                # 發送窗口完成進度
                if self.progress_callback:
                    self.progress_callback(i + 1, num_windows, f"窗口 {i+1} 完成")
                
            except Exception as e:
                print(f"  ❌ 處理失敗: {e}")
                if self.progress_callback:
                    self.progress_callback(i + 1, num_windows, f"窗口 {i+1} 失敗")
                continue
        
        total_time = time.time() - start_time
        
        # 4. 輸出總結
        print("\n" + "=" * 70)
        print("🎉 處理完成！")
        print("=" * 70)
        print(f"總耗時: {total_time:.2f}s")
        print(f"處理窗口數: {len(all_results)}/{num_windows}")
        if len(all_results) > 0:
            print(f"平均每窗口: {total_time/len(all_results):.2f}s")
        else:
            print("平均每窗口: N/A (無成功處理窗口)")
        
        # 5. 保存結果
        if save_results:
            output_file = Path(video_path).stem + "_results.json"
            output_path = Path("outputs") / output_file
            output_path.parent.mkdir(exist_ok=True)
            
            result_data = {
                'video_path': str(video_path),
                'video_name': Path(video_path).name,
                'total_frames': total_frames,
                'duration': round(total_frames / self.TARGET_FPS, 2),
                'num_windows': len(all_results),
                'window_size': self.WINDOW_SIZE,
                'stride': self.stride,
                'processing_time': round(total_time, 2),
                'results': all_results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 結果已保存: {output_path}")
        
        return all_results
    
    def compose_sentence_with_openai(self, results, target_language='繁體中文'):
        """
        使用 OpenAI 從多個窗口的 Top-5 結果中組合出最有可能的句子
        
        Args:
            results: 所有窗口的辨識結果
            target_language: 目標語言（繁體中文、英文、日文、韓文等）
        
        Returns:
            composed_sentence: 重組後的句子
            explanation: AI 的解釋
        """
        if not self.openai_client:
            # 如果沒有 OpenAI，返回 Top-1 單詞序列
            words = [result['top5'][0]['word'] for result in results]
            sentence = ' '.join(words)
            return sentence, "使用 Top-1 結果組合（未配置 OpenAI）"
        
        print("\n" + "=" * 70)
        print(f"🤖 使用 OpenAI 分析並重組句子（目標語言: {target_language}）...")
        print("=" * 70)
        
        # 發送 GPT 重組開始進度
        if self.progress_callback:
            self.progress_callback(0, 100, f"OpenAI 句子重組中 - 目標語言: {target_language}")
        
        # 準備提示詞
        prompt = self._build_openai_prompt(results, target_language)
        print(f"🔍 調試: 提示詞長度: {len(prompt)} 字符")
        print(f"🔍 調試: 窗口數量: {len(results)}")
        
        # 發送提示詞準備完成進度
        if self.progress_callback:
            self.progress_callback(30, 100, "提示詞準備完成 - 調用 GPT API")
        
        try:
            # 使用 OpenAI Chat Completions API（GPT-4o-mini）
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 使用穩定可靠的 GPT-4o-mini
                messages=[
                    {
                        "role": "system",
                        "content": """你是一個手語識別結果分析專家。你的任務是從多個或單個時間窗口的手語識別結果中，為每個窗口選出一個最合理的單詞，然後組合成有意義的句子，並以使用者的角度發想「我」，句子要合理，不能出現矛盾。

關鍵規則：
1. **每個窗口只選一個單詞**（從 Top-3 中挑選，不一定是 Top-1）
2. 選擇依據：信心度（通常正確的單詞 ≥ 10%）+ 語義合理性
3. 去除重複單詞（相鄰窗口可能識別同一個手勢）
4. 組合成符合手語語法的流暢句子，句子要合理，不能出現矛盾（可能與口語順序不同）
5. 如果某個單詞在多個連續窗口中高信心度出現，只保留一次

請為每個窗口選出一個單詞，然後組合成句子，若只有一個窗口，照樣選一個單詞組合成句子，組合成句子後不得重複從其他已選擇的單詞窗口選擇單詞。"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # 適中的溫度以獲得穩定結果
                max_tokens=1000  # GPT-4o-mini 使用 max_tokens
            )
            
            # 發送 API 調用完成進度
            if self.progress_callback:
                self.progress_callback(70, 100, "GPT API 回應成功 - 解析結果")
            
            # 解析 Chat Completions API 的回應
            ai_response = response.choices[0].message.content
            
            # 嘗試從回應中提取句子和解釋
            composed_sentence, explanation = self._parse_openai_response(ai_response)
            
            # 發送完成進度
            if self.progress_callback:
                self.progress_callback(100, 100, "句子重組完成")
            
            print(f"\n✅ AI 分析完成")
            print(f"🎯 重組句子: {composed_sentence}")
            print(f"💡 AI 解釋:\n{explanation}")
            
            return composed_sentence, explanation
            
        except Exception as e:
            print(f"❌ OpenAI API 調用失敗: {e}")
            return None, None
    
    def _build_openai_prompt(self, results, target_language='繁體中文'):
        """構建給 OpenAI 的提示詞"""
        prompt = "以下是手語識別系統對一段影片的分析結果，共有 {} 個時間窗口：\n\n".format(len(results))
        
        for result in results:
            window_id = result['window_id'] + 1
            time_range = f"{result['time_start']:.2f}s - {result['time_end']:.2f}s"
            
            # 添加明確的窗口開始標記（減少 GPT 混淆）
            prompt += f"【窗口 {window_id} 開始】({time_range})\n"
            for i, item in enumerate(result['top5'], 1):
                word = item['word']
                conf = item['confidence'] * 100
                prompt += f"  排名 {i}: {word:12s} 信心度 {conf:5.2f}%\n"
            prompt += f"【窗口 {window_id} 結束】\n\n"
        
        prompt += f"""請分析以上結果，完成以下任務：

【優先級規則】（按重要性遞減排序）
🔴 最高優先級：每個窗口必須選擇 Top-5 中的**一個**詞（不可跳過任何窗口）
🟠 次高優先級：選擇能與已選詞彙形成流暢、有意義句子的詞（語義連貫性）
🟡 中等優先級：優先選擇信心度 ≥ 10% 的詞（可靠性）
🟢 低優先級：去除相鄰窗口的重複詞

【詳細指示】
1. 逐窗口順序分析（窗口1 → 窗口2 → 窗口3 ...）
2. 對於每個窗口：
   - 首先檢查 Top-1 詞是否與已選詞重複
   - 如果不重複且信心度 ≥ 10%，優先選擇 Top-1
   - 如果重複或信心度 < 10%，選擇 Top-2 到 Top-5 中「最符合句意」的詞
3. 以使用者「我」的角度，組合成流暢、有意義的句子
4. 最終句子必須翻譯成{target_language}（保持手語原意）

【輸出格式】（嚴格遵循此格式）
句子：[重組後的{target_language}句子]
解釋：
- 窗口1: 選擇「[詞]」，信心度[X.XX%]，因為...
- 窗口2: 選擇「[詞]」，信心度[X.XX%]，因為...
- ...（逐一說明每個窗口的選擇理由）
- 最終組合邏輯：[說明如何處理重複詞、調整順序、驗證語義]"""
        
        return prompt
    
    def _parse_openai_response(self, response):
        """解析 OpenAI 的回應"""
        lines = response.strip().split('\n')
        sentence = ""
        explanation = ""
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("句子：") or line.startswith("句子:"):
                sentence = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                current_section = "sentence"
            elif line.startswith("解釋：") or line.startswith("解釋:"):
                explanation = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                current_section = "explanation"
            elif current_section == "explanation" and line:
                explanation += "\n" + line
        
        # 如果沒有找到標準格式，嘗試直接提取
        if not sentence:
            # 假設第一行是句子
            sentence = lines[0] if lines else response
        
        if not explanation:
            explanation = response
        
        return sentence, explanation.strip()
    
    def visualize_results(self, results, video_path=None):
        """
        視覺化結果（簡單的文字表格）
        
        Args:
            results: 處理結果
            video_path: 原始影片路徑（可選）
        """
        print("\n" + "=" * 70)
        print("📊 辨識結果總覽")
        print("=" * 70)
        
        # 統計最常出現的單詞（Top-1）
        word_counts = {}
        for result in results:
            top1_word = result['top5'][0]['word']
            word_counts[top1_word] = word_counts.get(top1_word, 0) + 1
        
        print("\nTop-1 單詞統計:")
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {word:12s}: {count} 次")
        
        print("\n時間軸結果:")
        print(f"{'窗口':<6} {'時間範圍':<15} {'Top-1 單詞':<12} {'信心度':<8} Top-2 ~ Top-5")
        print("-" * 70)
        
        for result in results:
            window_id = result['window_id'] + 1
            time_range = f"{result['time_start']:.1f}s-{result['time_end']:.1f}s"
            top1 = result['top5'][0]
            top_others = ", ".join([f"{r['word']}({r['confidence']*100:.0f}%)" 
                                   for r in result['top5'][1:]])
            
            print(f"{window_id:<6} {time_range:<15} {top1['word']:<12} "
                  f"{top1['confidence']*100:>5.1f}%   {top_others}")


def main():
    """主函數"""
    # 硬編碼參數
    video_path = "1.MOV"  # 輸入影片路徑
    model_path = 'model_output/best_model_mps.keras'  # 模型路徑
    label_path = 'model_output/label_map.json'  # 標籤映射路徑
    device = 'mps'  # 特徵提取設備
    stride = 60  # 滑動步長（幀數）- 25% 重疊，平衡精度和速度（方案 C）
    save_results = False  # 是否保存結果到 JSON（預設不保存）
    
    # OpenAI API Key（請設置您的 API Key）
    openai_api_key = os.environ.get('OPENAI_API_KEY')  # 從環境變數讀取
    
    # 檢查文件
    video_path_obj = Path(video_path)
    model_path_obj = Path(model_path)
    label_path_obj = Path(label_path)
    
    if not video_path_obj.exists():
        print(f"❌ 影片不存在: {video_path_obj}")
        return
    
    if not model_path_obj.exists():
        print(f"❌ 模型不存在: {model_path_obj}")
        return
    
    if not label_path_obj.exists():
        print(f"❌ 標籤映射不存在: {label_path_obj}")
        return
    
    # 創建識別器
    recognizer = SlidingWindowInference(
        model_path=model_path_obj,
        label_map_path=label_path_obj,
        device=device,
        stride=stride,
        openai_api_key=openai_api_key
    )
    
    # 處理影片
    results = recognizer.process_video(
        video_path=video_path_obj,
        save_results=save_results
    )
    
    # 視覺化結果
    recognizer.visualize_results(results, video_path_obj)
    
    # 使用 OpenAI 重組句子
    if openai_api_key:
        composed_sentence, explanation = recognizer.compose_sentence_with_openai(results)
        
        # 如果成功重組，保存到結果中
        if composed_sentence:
            print("\n" + "=" * 70)
            print("📝 最終結果")
            print("=" * 70)
            print(f"重組句子: {composed_sentence}")
            print("=" * 70)
    else:
        print("\n⚠️  未設置 OPENAI_API_KEY 環境變數，跳過句子重組功能")
        print("💡 提示：請設置環境變數 'export OPENAI_API_KEY=your-api-key' 或在代碼中硬編碼")


if __name__ == "__main__":
    main()

