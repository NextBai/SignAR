#!/usr/bin/env python3
"""
手語識別推論腳本 - M1 MPS 優化版本

⚠️  模型版本：v3.0 - Deep Improvements（深度改進版）

🎯 v3.0 關鍵改進：
1. 多尺度池化：GlobalMaxPool + GlobalAvgPool 融合
2. MaxNorm 約束：防止權重過大導致極端 logits
3. Focal Loss：關注難分類樣本，減少簡單樣本影響
4. Temperature Scaling：校準輸出信心度（T=1.5）
5. Mixup 數據增強：batch 級別樣本混合（僅訓練時）
6. Label Smoothing：軟化標籤分佈（ε=0.1）

📊 預期改善：
- 信心度：從 98%+ → 70-85%（更合理）
- 準確率：從 100% → 92-96%（健康）
- Top-1 到 Top-5 分佈更平滑

🔧 推論特性：
- 自動處理 BatchNormalization/LayerNormalization（推論模式）
- Temperature Scaling 層自動生效
- MaxNorm 約束在推論時不影響性能
- 無需任何代碼修改，直接使用訓練好的模型

整合流程：
1. 視頻標準化 (processor.py)
2. RGB 特徵提取 (MobileNetV3)
3. 骨架特徵提取 (MediaPipe Holistic - 159維)
4. Bi-GRU 推論（帶溫度校準）

支持：
- M1 MPS 加速（特徵提取）
- CPU 推論（BiGRU 在 CPU 上更快）
- 批次推論
- 模型精度轉換（bfloat16 → float32）
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json

# 設置 Keras backend 為 TensorFlow（M1 MPS 兼容）
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras

# 導入特徵提取器
sys.path.append(str(Path(__file__).parent / "feature_extraction"))


# ==================== 自定義層定義（載入模型需要）====================
@keras.saving.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    """Focal Loss - 用於載入訓練好的模型"""
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


class SignLanguageInference:
    """手語識別推論器"""
    
    def __init__(self, model_path, label_map_path, device='mps'):
        """
        初始化推論器
        
        Args:
            model_path: Keras 模型路徑 (.keras)
            label_map_path: 標籤映射 JSON 路徑
            device: 設備類型 ('mps', 'gpu', 'cpu')
        """
        self.device = device
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        
        # 設置 TensorFlow GPU/MPS
        self._setup_tf_device()
        
        # 載入模型和標籤
        self.model = self._load_model()
        self.label_map = self._load_label_map()
        self.idx_to_word = {v: k for k, v in self.label_map.items()}
        
        # 初始化特徵提取器
        self._init_extractors()
        
        print(f"✅ 推論器初始化完成！支持 {len(self.label_map)} 個手語單詞")
    
    def _setup_tf_device(self):
        """設置 TensorFlow 設備"""
        if self.device == 'mps':
            # M1 MPS 設置
            try:
                # TensorFlow Metal 插件會自動使用 MPS
                physical_devices = tf.config.list_physical_devices('GPU')
                if physical_devices:
                    print(f"🚀 使用 M1 MPS 加速（{len(physical_devices)} 個 GPU）")
                else:
                    print("⚠️  未檢測到 MPS，使用 CPU")
                    self.device = 'cpu'
            except Exception as e:
                print(f"⚠️  MPS 設置失敗: {e}，使用 CPU")
                self.device = 'cpu'
        elif self.device == 'gpu':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"🚀 使用 CUDA GPU（{len(gpus)} 個）")
                except RuntimeError as e:
                    print(f"⚠️  GPU 設置失敗: {e}")
        else:
            print("💻 使用 CPU")
    
    def _load_model(self):
        """載入 Keras 模型並轉換精度"""
        print(f"📥 載入模型: {self.model_path}")
        
        try:
            # 啟用 unsafe deserialization（處理 Lambda 層）
            keras.config.enable_unsafe_deserialization()
            
            # 載入模型（Keras 3 會自動處理 backend 差異）
            custom_objects = {
                'FocalLoss': FocalLoss,
                'MixupAccuracy': MixupAccuracy,
                'MixupTop3Accuracy': MixupTop3Accuracy
            }
            model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
            
            print(f"✅ 模型載入成功")
            print(f"   輸入形狀: {model.input_shape}")
            print(f"   輸出形狀: {model.output_shape}")
            
            # 檢查混合精度，如果是 bfloat16 則轉換為 float32
            if hasattr(model, 'dtype_policy'):
                print(f"   原始精度: {model.dtype_policy}")
            
            # 在 M1 MPS 上強制使用 float32
            if self.device == 'mps':
                print("🔧 轉換模型精度為 float32（M1 MPS 兼容）")
                # Keras 3 模型會自動處理精度
                keras.mixed_precision.set_global_policy('float32')
            
            return model
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise
    
    def _load_label_map(self):
        """載入標籤映射"""
        print(f"📋 載入標籤映射: {self.label_map_path}")
        
        with open(self.label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        
        print(f"✅ 載入 {len(label_map)} 個標籤")
        return label_map
    
    def _init_extractors(self):
        """初始化特徵提取器"""
        print("\n🔧 初始化特徵提取器...")
        
        # RGB 特徵提取器（PyTorch - MPS 加速）
        from rgb_feature_extraction import RGBFeatureExtractor
        
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

        # MediaPipe Holistic 自動優化
        self.skeleton_extractor = EnhancedSkeletonExtractor(num_threads=4)
        
        print("✅ 特徵提取器初始化完成")
    
    def extract_features(self, video_path, max_length=300):
        """
        提取視頻特徵（RGB + Skeleton）
        
        Args:
            video_path: 視頻路徑
            max_length: 最大序列長度
        
        Returns:
            features: (max_length, 1011) 的特徵向量
        """
        video_path = Path(video_path)
        
        # 1. 提取 RGB 特徵 (T, 960)
        rgb_features = self.rgb_extractor.extract_features(video_path)
        if rgb_features is None:
            raise ValueError(f"無法提取 RGB 特徵: {video_path}")

        # 2. 提取骨架特徵 (T, 159) - MediaPipe Holistic
        skeleton_features = self.skeleton_extractor.extract_features(video_path)
        if skeleton_features is None:
            raise ValueError(f"無法提取骨架特徵: {video_path}")

        # 3. 對齊長度
        min_len = min(len(rgb_features), len(skeleton_features))
        rgb_features = rgb_features[:min_len]
        skeleton_features = skeleton_features[:min_len]

        # 4. Concat 融合 (T, 1119)
        concat_features = np.concatenate([rgb_features, skeleton_features], axis=1)

        # 5. Padding 或截斷到固定長度
        if len(concat_features) < max_length:
            padding = np.zeros((max_length - len(concat_features), 1119), dtype=np.float32)
            concat_features = np.concatenate([concat_features, padding], axis=0)
        else:
            concat_features = concat_features[:max_length]
        
        return concat_features
    
    def predict(self, video_path, top_k=3):
        """
        預測單個視頻
        
        Args:
            video_path: 視頻路徑
            top_k: 返回前 k 個預測結果
        
        Returns:
            predictions: [(word, confidence), ...] 按信心度排序
        """
        # 提取特徵
        features = self.extract_features(video_path)
        
        # 擴展為 batch (1, max_length, 1011)
        features_batch = np.expand_dims(features, axis=0)
        
        # 推論
        with tf.device('/GPU:0' if self.device in ['mps', 'gpu'] else '/CPU:0'):
            predictions = self.model.predict(features_batch, verbose=0)
        
        # 獲取 top-k 結果
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            word = self.idx_to_word[idx]
            confidence = float(predictions[0][idx])
            results.append((word, confidence))
        
        return results
    
    def predict_batch(self, video_paths, batch_size=8, top_k=3):
        """
        批次預測多個視頻
        
        Args:
            video_paths: 視頻路徑列表
            batch_size: 批次大小
            top_k: 返回前 k 個預測結果
        
        Returns:
            all_results: [[(word, confidence), ...], ...]
        """
        all_results = []
        
        for i in tqdm(range(0, len(video_paths), batch_size), desc="批次推論"):
            batch_paths = video_paths[i:i + batch_size]
            batch_features = []
            
            # 提取特徵
            for video_path in batch_paths:
                try:
                    features = self.extract_features(video_path)
                    batch_features.append(features)
                except Exception as e:
                    print(f"⚠️  跳過 {video_path}: {e}")
                    batch_features.append(np.zeros((300, 1119), dtype=np.float32))
            
            # 批次推論
            batch_features = np.array(batch_features)  # (batch, 300, 1119)
            
            with tf.device('/GPU:0' if self.device in ['mps', 'gpu'] else '/CPU:0'):
                predictions = self.model.predict(batch_features, verbose=0)
            
            # 解析結果
            for pred in predictions:
                top_indices = np.argsort(pred)[::-1][:top_k]
                results = [(self.idx_to_word[idx], float(pred[idx])) for idx in top_indices]
                all_results.append(results)
        
        return all_results


def demo_single_video():
    """單個視頻推論示範"""
    print("=" * 70)
    print("🎯 手語識別推論 - 單個視頻示範")
    print("=" * 70)
    
    # 配置路徑
    model_dir = Path("/Users/baidongqu/Desktop/MVP/model_output")
    model_path = model_dir / "best_model_mps.keras"
    label_map_path = model_dir / "label_map.json"
    
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return
    
    # 初始化推論器
    inferencer = SignLanguageInference(
        model_path=model_path,
        label_map_path=label_map_path,
        device='mps'  # M1 Mac 使用 'mps'，CUDA GPU 使用 'gpu'，否則 'cpu'
    )
    
    # 測試視頻
    test_video = Path("/Users/baidongqu/Desktop/MVP/1.MP4")
    
    if not test_video.exists():
        print(f"❌ 測試視頻不存在: {test_video}")
        return
    
    print(f"\n🎬 測試視頻: {test_video.name}")
    print("🔄 開始推論...")
    
    # 推論
    results = inferencer.predict(test_video, top_k=5)
    
    # 輸出結果
    print("\n" + "=" * 70)
    print("🎉 推論結果：")
    print("=" * 70)
    for i, (word, confidence) in enumerate(results, 1):
        bar = "█" * int(confidence * 50)
        print(f"{i}. {word:15s} {confidence*100:6.2f}% {bar}")
    print("=" * 70)


def demo_batch_inference():
    """批次推論示範"""
    print("=" * 70)
    print("🎯 手語識別推論 - 批次處理示範")
    print("=" * 70)
    
    # 配置路徑
    model_dir = Path("/Users/baidongqu/Desktop/MVP/model_output")
    model_path = model_dir / "best_model_mps.keras"
    label_map_path = model_dir / "label_map.json"
    
    # 初始化推論器
    inferencer = SignLanguageInference(
        model_path=model_path,
        label_map_path=label_map_path,
        device='mps'
    )
    
    # 獲取測試視頻（每個類別選 3 個）
    videos_dir = Path("/Users/baidongqu/Desktop/MVP/videos")
    test_videos = []
    ground_truth = []
    
    for word_dir in sorted(videos_dir.iterdir()):
        if word_dir.is_dir():
            videos = list(word_dir.glob("*.mp4"))[:3]  # 每個類別 3 個視頻
            test_videos.extend(videos)
            ground_truth.extend([word_dir.name] * len(videos))
    
    print(f"\n📊 測試集: {len(test_videos)} 個視頻")
    
    # 批次推論
    results = inferencer.predict_batch(test_videos, batch_size=4, top_k=3)
    
    # 統計準確率
    correct = 0
    top3_correct = 0
    
    print("\n" + "=" * 70)
    print("📋 詳細結果：")
    print("=" * 70)
    
    for i, (video_path, gt, preds) in enumerate(zip(test_videos, ground_truth, results)):
        top1_word, top1_conf = preds[0]
        
        if top1_word == gt:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        if any(word == gt for word, _ in preds):
            top3_correct += 1
        
        print(f"{status} {video_path.name:25s} | GT: {gt:10s} | Pred: {top1_word:10s} ({top1_conf*100:.1f}%)")
    
    # 輸出統計
    print("\n" + "=" * 70)
    print("📊 統計結果：")
    print("=" * 70)
    print(f"Top-1 準確率: {correct}/{len(test_videos)} = {correct/len(test_videos)*100:.2f}%")
    print(f"Top-3 準確率: {top3_correct}/{len(test_videos)} = {top3_correct/len(test_videos)*100:.2f}%")
    print("=" * 70)


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='手語識別推論')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                       help='推論模式: single (單個視頻) 或 batch (批次處理)')
    parser.add_argument('--video', type=str, help='視頻路徑（single 模式）')
    parser.add_argument('--model', type=str, default='model_output/best_model.keras',
                       help='模型路徑')
    parser.add_argument('--device', type=str, default='mps', choices=['mps', 'gpu', 'cpu'],
                       help='設備類型')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if args.video:
            # 自定義視頻
            model_dir = Path(args.model).parent
            inferencer = SignLanguageInference(
                model_path=args.model,
                label_map_path=model_dir / "label_map.json",
                device=args.device
            )
            results = inferencer.predict(args.video, top_k=5)
            
            print("\n🎉 推論結果：")
            for i, (word, confidence) in enumerate(results, 1):
                print(f"{i}. {word}: {confidence*100:.2f}%")
        else:
            # 示範模式
            demo_single_video()
    else:
        demo_batch_inference()


if __name__ == "__main__":
    main()

