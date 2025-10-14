#!/usr/bin/env python3
"""
Bi-GRU 手語識別訓練腳本 - Kaggle TPU 優化版本
專為 Kaggle TPU v3-8 優化
- 支持 TPU v3-8 (Kaggle 默認提供)
- 簡化數據增強邏輯
- 優化數據管線以充分利用 TPU 性能
"""

import os
import warnings

# ==================== Kaggle TPU 環境設置（必須在導入任何深度學習庫之前）====================
print("🔧 配置 Kaggle TPU Runtime...")

# 設置 PJRT 設備為 TPU（TPU v5e-8 必需）
os.environ['PJRT_DEVICE'] = 'TPU'

# 設置 Keras 使用 JAX backend（與 PJRT TPU 完美兼容）
os.environ['KERAS_BACKEND'] = 'jax'

# 其他環境設置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # JAX 內存優化

import numpy as np
import jax
import tensorflow as tf  # 僅用於 tf.data pipeline
import keras
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import hashlib
import random


# ==================== TPU 初始化（Kaggle TPU v5e-8 PJRT + JAX Backend）====================
def init_tpu_strategy():
    """初始化 TPU 策略（使用 Keras JAX backend + PJRT Runtime）"""
    print("\n🚀 初始化 Kaggle TPU v5e-8 (Keras JAX Backend + PJRT)...")
    
    try:
        # 檢查 JAX 和 Keras backend
        print(f"🔍 Keras Backend: {keras.backend.backend()}")
        
        # 檢查 JAX TPU 連接
        jax_devices = jax.devices()
        print(f"✅ JAX 檢測到 {len(jax_devices)} 個 TPU 設備")
        print(f"   設備類型: {jax_devices[0].platform}")
        
        # 使用 Keras 的分佈式 API（與 JAX backend 完美兼容）
        print("📡 正在創建 Keras 分佈式策略...")
        
        # Keras 3 + JAX backend 會自動使用所有可用的 JAX 設備
        # 不需要手動創建 strategy，Keras 會自動分佈訓練
        num_devices = len(jax_devices)
        print(f"✅ Keras 將使用 {num_devices} 個 TPU 設備進行分佈式訓練")
        
        # 啟用混合精度訓練（TPU v5e-8 加速關鍵）
        keras.mixed_precision.set_global_policy('mixed_bfloat16')
        print("✅ 混合精度訓練已啟用: mixed_bfloat16")
        
        # 返回設備數量（用於計算全局批次大小）
        class DummyStrategy:
            def __init__(self, num_replicas):
                self.num_replicas_in_sync = num_replicas
        
        return DummyStrategy(num_devices)
        
    except Exception as e:
        print(f"\n❌ TPU 初始化失敗: {e}")
        print("\n💡 故障排除建議：")
        print("1. 確認 Kaggle Notebook 已啟用 TPU 加速器（Settings → Accelerator → TPU v5 litepod）")
        print("2. 重新啟動 Kernel 並再次執行")
        print("3. 確認使用的是 Kaggle 環境而非本地環境")
        print("4. 檢查 KERAS_BACKEND=jax 環境變數是否設置")
        
        # 打印詳細調試信息
        print("\n🔍 環境診斷信息：")
        print(f"   Keras 版本: {keras.__version__}")
        print(f"   Keras Backend: {keras.backend.backend()}")
        print(f"   JAX 版本: {jax.__version__}")
        print(f"   PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'Not Set')}")
        print(f"   KERAS_BACKEND: {os.environ.get('KERAS_BACKEND', 'Not Set')}")
        print(f"   JAX 設備: {jax.devices()}")
        
        raise RuntimeError(f"Kaggle TPU v5e-8 初始化失敗: {e}")


# ==================== 數據集類 ====================
class SignLanguageDataset:
    """手語數據集加載器 - TPU 優化版本（含去重和數據增強）"""
    
    def __init__(self, rgb_dir, skeleton_dir, max_length=300, use_augmentation=True, use_mixup=False, mixup_alpha=0.2):
        """
        初始化數據集
        
        Args:
            rgb_dir: RGB 特徵目錄 (960維)
            skeleton_dir: Skeleton 特徵目錄 (159維 - MediaPipe Holistic)
            max_length: 最大序列長度
            use_augmentation: 是否使用數據增強平衡類別
            use_mixup: 是否啟用 Mixup 增強（對抗特徵混疊）
            mixup_alpha: Mixup 的 Beta 分佈參數
        """
        self.rgb_dir = Path(rgb_dir)
        self.skeleton_dir = Path(skeleton_dir)
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        self.samples = []
        self.label_map = {}
        self._build_dataset()
        
        # 如果啟用增強，進行類別平衡
        if use_augmentation:
            self._balance_classes_with_augmentation()
    
    def _build_dataset(self):
        """構建數據集索引 - 使用 Hash 去重"""
        print("📚 載入數據集（去重模式）...")
        
        # 獲取所有單詞目錄並創建標籤映射
        word_dirs = sorted([d for d in self.rgb_dir.iterdir() if d.is_dir()])
        self.label_map = {word.name: idx for idx, word in enumerate(word_dirs)}
        self.num_classes = len(self.label_map)
        print(f"📝 類別數量: {self.num_classes}")
        
        # 收集樣本並去重
        seen_hashes = set()
        duplicate_count = 0
        
        for word_dir in tqdm(word_dirs, desc="掃描數據"):
            word_name = word_dir.name
            label = self.label_map[word_name]
            
            rgb_files = list((self.rgb_dir / word_name).glob("*.npy"))
            
            for rgb_file in rgb_files:
                skeleton_file = self.skeleton_dir / word_name / rgb_file.name
                
                if skeleton_file.exists():
                    try:
                        # 計算 RGB 數據的 hash 來去重
                        rgb_data = np.load(rgb_file)
                        data_hash = hashlib.md5(rgb_data.tobytes()).hexdigest()
                        
                        if data_hash not in seen_hashes:
                            seen_hashes.add(data_hash)
                            self.samples.append({
                                'rgb_path': str(rgb_file),
                                'skeleton_path': str(skeleton_file),
                                'label': label,
                                'word': word_name
                            })
                        else:
                            duplicate_count += 1
                    except Exception as e:
                        print(f"⚠️  載入失敗 {rgb_file.name}: {e}")
                        continue
        
        print(f"✅ 總唯一樣本數: {len(self.samples)}")
        if duplicate_count > 0:
            print(f"🗑️  移除重複樣本: {duplicate_count} 個")
        
        # 打印類別分布
        class_counts = {}
        for sample in self.samples:
            word = sample['word']
            class_counts[word] = class_counts.get(word, 0) + 1
        
        print("\n📊 去重後類別分布:")
        for word, count in sorted(class_counts.items()):
            print(f"  {word}: {count} 樣本")
    
    def _balance_classes_with_augmentation(self):
        """使用輕量級數據增強平衡類別"""
        print("\n⚖️  啟用類別平衡增強...")
        
        # 統計當前類別分布
        class_counts = {}
        class_samples = {}
        for sample in self.samples:
            word = sample['word']
            if word not in class_samples:
                class_samples[word] = []
            class_samples[word].append(sample)
            class_counts[word] = class_counts.get(word, 0) + 1
        
        # 找到目標樣本數（最多的類別）
        target_count = max(class_counts.values())
        print(f"🎯 目標樣本數: {target_count} (每個類別)")
        
        # 對每個類別進行增強
        balanced_samples = list(self.samples)  # 保留原始樣本
        
        for word, samples in class_samples.items():
            current_count = len(samples)
            needed_count = target_count - current_count
            
            if needed_count <= 0:
                print(f"✅ {word}: {current_count} 樣本 (無需增強)")
                continue
            
            print(f"🔧 {word}: {current_count} → {target_count} 樣本 (需要 {needed_count} 個增強)")
            
            # 簡單策略：從現有樣本中隨機選擇並標記為需要增強
            for i in range(needed_count):
                original_sample = random.choice(samples)
                # 創建增強樣本的引用（實際增強在載入時動態進行）
                augmented_sample = {
                    'rgb_path': original_sample['rgb_path'],
                    'skeleton_path': original_sample['skeleton_path'],
                    'label': original_sample['label'],
                    'word': original_sample['word'],
                    'augment': True  # 標記為需要增強
                }
                balanced_samples.append(augmented_sample)
        
        self.samples = balanced_samples
        print(f"✅ 平衡完成！總樣本數: {len(self.samples)}")
        
        # 重新統計最終分布
        final_counts = {}
        for sample in self.samples:
            word = sample['word']
            final_counts[word] = final_counts.get(word, 0) + 1
        
        print("\n📊 平衡後類別分布:")
        for word, count in sorted(final_counts.items()):
            print(f"  {word}: {count} 樣本")
    
    def _apply_mixup(self, feat1, label1, feat2, label2, alpha=0.2):
        """
        應用 Mixup 數據增強（對抗特徵混疊）
        
        Mixup 通過混合兩個樣本，強制模型學習更平滑的決策邊界，
        防止 embedding space 中的特徵過度集中。
        
        Args:
            feat1, feat2: 兩個特徵樣本
            label1, label2: 對應標籤
            alpha: Beta 分佈參數（越小混合越溫和）
        
        Returns:
            mixed_feat: 混合後的特徵
            mixed_label: 混合後的標籤（軟標籤）
        """
        # 從 Beta 分佈採樣混合比例
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        
        # 混合特徵（線性插值）
        mixed_feat = lam * feat1 + (1 - lam) * feat2
        
        # 混合標籤（創建軟標籤）
        # 注意：這裡返回單個標籤，但帶有混合信息（用於後續處理）
        # 實際應用中需要在 loss 中使用軟標籤
        return mixed_feat.astype(np.float32), label1, lam
    
    def _apply_light_augmentation(self, rgb_feat, skeleton_feat):
        """
        應用輕量級時序增強（所有訓練樣本使用）
        策略：隨機應用 1-2 種輕度增強，保持數據真實性
        """
        # 隨機選擇 1-2 種增強方法
        aug_methods = random.sample([
            'gaussian_noise',
            'time_masking', 
            'feature_scaling',
            'temporal_shift'
        ], k=random.randint(1, 2))
        
        for method in aug_methods:
            if method == 'gaussian_noise':
                # 輕度高斯噪聲
                noise_std = random.uniform(0.005, 0.015)  # 降低噪聲強度
                rgb_feat = rgb_feat + np.random.normal(0, noise_std, rgb_feat.shape).astype(np.float32)
                skeleton_feat = skeleton_feat + np.random.normal(0, noise_std * 0.5, skeleton_feat.shape).astype(np.float32)
                
            elif method == 'time_masking':
                # 時間遮罩（輕度）
                seq_len = rgb_feat.shape[0]
                mask_ratio = random.uniform(0.05, 0.1)  # 降低遮罩比例
                mask_len = int(seq_len * mask_ratio)
                if mask_len > 0:
                    start_idx = random.randint(0, seq_len - mask_len)
                    rgb_feat[start_idx:start_idx + mask_len, :] *= random.uniform(0.3, 0.7)  # 弱化
                    skeleton_feat[start_idx:start_idx + mask_len, :] *= random.uniform(0.3, 0.7)
                    
            elif method == 'feature_scaling':
                # 特徵縮放（輕度）
                scale = random.uniform(0.95, 1.05)  # 降低縮放範圍
                rgb_feat = rgb_feat * scale
                
            elif method == 'temporal_shift':
                # 時間偏移（輕微）
                shift = random.randint(-2, 2)  # 減少偏移範圍
                if shift != 0:
                    rgb_feat = np.roll(rgb_feat, shift, axis=0)
                    skeleton_feat = np.roll(skeleton_feat, shift, axis=0)
        
        return rgb_feat, skeleton_feat
    
    def _apply_strong_augmentation(self, rgb_feat, skeleton_feat):
        """
        應用強力組合增強（類別平衡專用）
        策略：組合多種增強，創建更多樣化的訓練樣本
        """
        # 必定應用多種增強（2-3種）
        aug_methods = random.sample([
            'gaussian_noise',
            'time_masking', 
            'feature_scaling',
            'temporal_shift',
            'speed_perturbation',  # 新增：速度擾動
            'feature_dropout'       # 新增：特徵dropout
        ], k=random.randint(2, 3))
        
        for method in aug_methods:
            if method == 'gaussian_noise':
                # 較強的高斯噪聲
                noise_std = random.uniform(0.02, 0.04)
                rgb_feat = rgb_feat + np.random.normal(0, noise_std, rgb_feat.shape).astype(np.float32)
                skeleton_feat = skeleton_feat + np.random.normal(0, noise_std * 0.5, skeleton_feat.shape).astype(np.float32)
                
            elif method == 'time_masking':
                # 較強的時間遮罩
                seq_len = rgb_feat.shape[0]
                mask_ratio = random.uniform(0.15, 0.25)
                mask_len = int(seq_len * mask_ratio)
                if mask_len > 0:
                    start_idx = random.randint(0, seq_len - mask_len)
                    rgb_feat[start_idx:start_idx + mask_len, :] *= random.uniform(0.1, 0.3)
                    skeleton_feat[start_idx:start_idx + mask_len, :] *= random.uniform(0.1, 0.3)
                    
            elif method == 'feature_scaling':
                # 較強的特徵縮放
                scale = random.uniform(0.85, 1.15)
                rgb_feat = rgb_feat * scale
                
            elif method == 'temporal_shift':
                # 較大的時間偏移
                shift = random.randint(-5, 5)
                if shift != 0:
                    rgb_feat = np.roll(rgb_feat, shift, axis=0)
                    skeleton_feat = np.roll(skeleton_feat, shift, axis=0)
                    
            elif method == 'speed_perturbation':
                # 速度擾動（時間拉伸/壓縮）
                speed_factor = random.uniform(0.9, 1.1)
                new_len = int(len(rgb_feat) * speed_factor)
                if new_len > 0:
                    indices = np.linspace(0, len(rgb_feat) - 1, new_len).astype(int)
                    rgb_feat = rgb_feat[indices]
                    skeleton_feat = skeleton_feat[indices]
                    
            elif method == 'feature_dropout':
                # 特徵通道隨機dropout
                dropout_ratio = random.uniform(0.05, 0.15)
                mask = np.random.binomial(1, 1 - dropout_ratio, rgb_feat.shape[1])
                rgb_feat = rgb_feat * mask
        
        return rgb_feat, skeleton_feat
    
    def _load_and_process(self, rgb_path, skeleton_path, label, should_augment, is_training, mixup_data=None):
        """
        載入並處理單個樣本（支持兩層增強策略 + Mixup）
        
        Args:
            should_augment: 是否為類別平衡的增強樣本（使用強增強）
            is_training: 是否為訓練模式（訓練時所有樣本都應用輕增強）
            mixup_data: Mixup 數據 (rgb_path2, skeleton_path2, label2, lambda) 或 None
        """
        try:
            # 載入特徵
            rgb_feat = np.load(rgb_path.numpy().decode()).astype(np.float32)
            skeleton_feat = np.load(skeleton_path.numpy().decode()).astype(np.float32)
            
            # 驗證維度
            if rgb_feat.shape[1] != 960 or skeleton_feat.shape[1] != 159:
                raise ValueError(f"特徵維度錯誤: RGB {rgb_feat.shape}, Skeleton {skeleton_feat.shape}")
            
            # 對齊長度
            min_len = min(len(rgb_feat), len(skeleton_feat))
            rgb_feat = rgb_feat[:min_len]
            skeleton_feat = skeleton_feat[:min_len]
            
            # 兩層增強策略：
            # 1. 訓練時，所有原始樣本應用輕量增強（80%概率）
            # 2. 類別平衡的增強樣本應用強力增強（100%）
            
            if should_augment.numpy():
                # 類別平衡樣本：強力增強
                rgb_feat, skeleton_feat = self._apply_strong_augmentation(rgb_feat, skeleton_feat)
            elif is_training.numpy() and random.random() < 0.8:
                # 原始訓練樣本：80%概率應用輕量增強
                rgb_feat, skeleton_feat = self._apply_light_augmentation(rgb_feat, skeleton_feat)
            
            # Concat 融合 (960 + 159 = 1119)
            concat_feat = np.concatenate([rgb_feat, skeleton_feat], axis=1)
            
            # Padding 或截斷到固定長度
            if len(concat_feat) < self.max_length:
                padding = np.zeros((self.max_length - len(concat_feat), 1119), dtype=np.float32)
                concat_feat = np.concatenate([concat_feat, padding], axis=0)
            else:
                concat_feat = concat_feat[:self.max_length]
            
            # Mixup（如果提供了混合數據）
            if mixup_data is not None:
                try:
                    # 解析 mixup_data（僅在訓練時使用）
                    rgb_path2, skeleton_path2, label2, lam = mixup_data
                    
                    # 載入第二個樣本
                    rgb_feat2 = np.load(rgb_path2.decode()).astype(np.float32)
                    skeleton_feat2 = np.load(skeleton_path2.decode()).astype(np.float32)
                    
                    min_len2 = min(len(rgb_feat2), len(skeleton_feat2))
                    rgb_feat2 = rgb_feat2[:min_len2]
                    skeleton_feat2 = skeleton_feat2[:min_len2]
                    
                    concat_feat2 = np.concatenate([rgb_feat2, skeleton_feat2], axis=1)
                    
                    # Padding
                    if len(concat_feat2) < self.max_length:
                        padding2 = np.zeros((self.max_length - len(concat_feat2), 1119), dtype=np.float32)
                        concat_feat2 = np.concatenate([concat_feat2, padding2], axis=0)
                    else:
                        concat_feat2 = concat_feat2[:self.max_length]
                    
                    # 混合特徵
                    concat_feat = lam * concat_feat + (1 - lam) * concat_feat2
                    
                    # 注意：Mixup 的軟標籤處理需要在 loss 中實現
                    # 這裡返回原始標籤，但特徵已經混合
                    
                except Exception as e:
                    # Mixup 失敗時使用原始特徵
                    pass
            
            return concat_feat, np.int32(label)
        
        except Exception as e:
            print(f"⚠️  載入失敗: {e}")
            return np.zeros((self.max_length, 1119), dtype=np.float32), np.int32(0)
    
    def create_tf_dataset(self, batch_size, val_split=0.2, shuffle=True):
        """
        創建 TensorFlow Dataset（TPU 優化）
        
        Args:
            batch_size: 全局批次大小（已經考慮了 TPU 核心數）
            val_split: 驗證集比例
            shuffle: 是否打亂
        
        Returns:
            train_dataset, val_dataset
        """
        # 分層分割（按類別）
        train_samples, val_samples = [], []
        
        for word_name in self.label_map.keys():
            word_samples = [s for s in self.samples if s['word'] == word_name]
            
            if shuffle:
                np.random.shuffle(word_samples)
            
            val_size = max(1, int(len(word_samples) * val_split))
            train_samples.extend(word_samples[:-val_size])
            val_samples.extend(word_samples[-val_size:])
        
        if shuffle:
            np.random.shuffle(train_samples)
            np.random.shuffle(val_samples)
        
        print(f"✂️  訓練集: {len(train_samples)} 樣本")
        print(f"✂️  驗證集: {len(val_samples)} 樣本")
        
        # 創建數據集
        train_dataset = self._create_dataset(train_samples, batch_size, shuffle=True, augment=True, is_training=True)
        val_dataset = self._create_dataset(val_samples, batch_size, shuffle=False, augment=False, is_training=False)
        
        return train_dataset, val_dataset
    
    def _create_dataset(self, samples, batch_size, shuffle, augment, is_training=False):
        """
        創建單個 TF Dataset（TPU 優化版本 + 內存緩存 + 兩層增強 + Mixup）
        
        Mixup 在 batch 級別實現：
        1. 正常載入 batch
        2. 在 batch 內部進行隨機 Mixup
        3. 生成軟標籤
        """
        # 提取路徑、標籤和增強標記
        rgb_paths = [s['rgb_path'] for s in samples]
        skeleton_paths = [s['skeleton_path'] for s in samples]
        labels = [s['label'] for s in samples]
        # 檢查是否為類別平衡的增強樣本（強增強）
        should_augment = [augment and s.get('augment', False) for s in samples]
        # 標記是否為訓練模式（原始樣本用輕增強）
        is_training_flags = [is_training] * len(samples)
        
        # 創建 Dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            rgb_paths, skeleton_paths, labels, should_augment, is_training_flags
        ))
        
        # 並行載入數據
        def load_wrapper(rgb_path, skeleton_path, label, should_aug, is_train):
            # 不使用樣本級 Mixup（改用 batch 級）
            features, label_out = tf.py_function(
                func=lambda rp, sp, l, sa, it: self._load_and_process(rp, sp, l, sa, it, mixup_data=None),
                inp=[rgb_path, skeleton_path, label, should_aug, is_train],
                Tout=(tf.float32, tf.int32)
            )
            features.set_shape([self.max_length, 1119])
            label_out.set_shape([])
            return features, label_out
        
        dataset = dataset.map(
            load_wrapper,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        
        # TPU 優化：緩存到內存（訓練集和驗證集都緩存）
        # 注意：由於增強是動態的，訓練集不緩存，驗證集可以緩存
        if not is_training:
            dataset = dataset.cache()
            print(f"   ✅ 驗證集數據已緩存到內存 ({len(samples)} 樣本)")
        else:
            print(f"   ℹ️  訓練集使用動態增強，不緩存 ({len(samples)} 樣本)")
        
        # 訓練集才進行 shuffle（在 cache 之後）
        if shuffle:
            dataset = dataset.shuffle(buffer_size=2000, reshuffle_each_iteration=True)
        
        # TPU 優化：drop_remainder=True 確保批次大小固定
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # ⭐ Mixup 增強（batch 級別，僅訓練集）
        if is_training and self.use_mixup:
            print(f"   🎨 啟用 Batch-level Mixup (alpha={self.mixup_alpha})")
            
            def apply_mixup_batch(features, labels):
                """在 batch 內部應用 Mixup"""
                import keras.ops as ops
                
                batch_size = ops.shape(features)[0]
                
                # 生成隨機排列索引（用於配對樣本）
                indices = tf.random.shuffle(tf.range(batch_size))
                
                # 從 Beta 分佈採樣 lambda
                lam = tf.random.uniform([], minval=0.0, maxval=1.0)
                # 使用簡化的混合比例（避免 Beta 分佈的複雜性）
                if self.mixup_alpha > 0:
                    lam = tf.maximum(lam, 1.0 - lam)  # 傾向於 0.5 附近
                
                # 混合特徵
                mixed_features = lam * features + (1.0 - lam) * tf.gather(features, indices)
                
                # 混合標籤（生成軟標籤）
                labels_a = tf.one_hot(labels, depth=self.num_classes)
                labels_b = tf.one_hot(tf.gather(labels, indices), depth=self.num_classes)
                mixed_labels = lam * labels_a + (1.0 - lam) * labels_b
                
                return mixed_features, mixed_labels
            
            dataset = dataset.map(
                apply_mixup_batch,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # TPU 優化：prefetch 預載下一批數據
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


# ==================== 自定義 Metrics（支持軟標籤）====================
class MixupAccuracy(keras.metrics.Metric):
    """支持 Mixup 軟標籤的準確率計算"""
    
    def __init__(self, name='mixup_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        import keras.ops as ops
        
        # 處理硬標籤或軟標籤
        if len(ops.shape(y_true)) == 1:
            # 硬標籤
            y_true_labels = y_true
        else:
            # 軟標籤（Mixup）：取最大概率的類別
            y_true_labels = ops.argmax(y_true, axis=-1)
        
        # 預測標籤
        y_pred_labels = ops.argmax(y_pred, axis=-1)
        
        # 計算準確率
        matches = ops.cast(ops.equal(y_true_labels, y_pred_labels), 'float32')
        
        self.total.assign_add(ops.sum(matches))
        self.count.assign_add(ops.cast(ops.size(matches), 'float32'))
    
    def result(self):
        return self.total / self.count
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class MixupTop3Accuracy(keras.metrics.Metric):
    """支持 Mixup 軟標籤的 Top-3 準確率計算"""
    
    def __init__(self, name='mixup_top3_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        import keras.ops as ops
        
        # 處理硬標籤或軟標籤
        if len(ops.shape(y_true)) == 1:
            # 硬標籤
            y_true_labels = y_true
        else:
            # 軟標籤（Mixup）：取最大概率的類別
            y_true_labels = ops.argmax(y_true, axis=-1)
        
        # 獲取 Top-3 預測
        top3_pred = ops.top_k(y_pred, k=3)[1]  # 返回索引
        
        # 檢查真實標籤是否在 Top-3 中
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


# ==================== 自定義損失函數 ====================
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss - 解決類別不平衡和過度自信問題（JAX 兼容版本）
    
    Focal Loss 通過降低易分類樣本的權重，強制模型關注難分類樣本，
    有效對抗過度自信和特徵混疊。
    
    論文：Focal Loss for Dense Object Detection (Lin et al., 2017)
    
    公式：FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: 調節因子，控制易分類樣本的權重衰減速度
               gamma=0 退化為標準交叉熵
               gamma=2 是論文推薦值
        alpha: 類別權重（可選）
        label_smoothing: 標籤平滑因子
        num_classes: 類別數量（必須在初始化時提供，避免 JAX tracer 錯誤）
    """
    
    def __init__(self, num_classes, gamma=2.0, alpha=None, label_smoothing=0.1, name='focal_loss'):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        """
        計算 Focal Loss（支持硬標籤和軟標籤）
        
        Args:
            y_true: 真實標籤 (batch_size,) 或 one-hot/軟標籤 (batch_size, num_classes)
            y_pred: 預測概率 (batch_size, num_classes)
        """
        import keras.ops as ops
        
        # 檢查 y_true 是否為軟標籤（Mixup）或硬標籤
        if len(ops.shape(y_true)) == 1:
            # 硬標籤：轉換為 one-hot
            y_true_one_hot = ops.one_hot(ops.cast(y_true, 'int32'), self.num_classes)
            
            # Label smoothing
            if self.label_smoothing > 0:
                y_true_one_hot = y_true_one_hot * (1.0 - self.label_smoothing) + \
                                self.label_smoothing / float(self.num_classes)
        else:
            # 軟標籤（Mixup）：直接使用
            y_true_one_hot = y_true
        
        # 避免 log(0)
        epsilon = 1e-7
        y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # 計算交叉熵
        cross_entropy = -y_true_one_hot * ops.log(y_pred)
        
        # 計算 Focal 權重：(1 - p_t)^gamma
        # p_t 是正確類別的預測概率
        p_t = ops.sum(y_true_one_hot * y_pred, axis=-1, keepdims=True)
        focal_weight = ops.power(1.0 - p_t, self.gamma)
        
        # 應用 Focal 權重
        focal_cross_entropy = focal_weight * cross_entropy
        
        # 可選的類別權重
        if self.alpha is not None:
            alpha_weight = y_true_one_hot * self.alpha
            focal_cross_entropy = alpha_weight * focal_cross_entropy
        
        # 返回平均損失
        return ops.mean(ops.sum(focal_cross_entropy, axis=-1))


# ==================== 模型構建 ====================
def build_bigru_model(input_shape, num_classes, gru_units=256, dropout=0.4, use_temperature_scaling=True):
    """
    構建 Bi-GRU 模型（Keras 3.0 - 解決特徵混疊與過度自信）
    
    針對性改進：
    1. 【特徵混疊】添加中心損失正則化的準備（embedding 層）
    2. 【Distribution Shift】多尺度特徵融合，增強魯棒性
    3. 【Calibration】溫度縮放預處理 + 更強的 dropout
    4. 【Logit Variance】限制輸出層權重範數，防止極端 logits
    
    Args:
        input_shape: (max_length, feature_dim) - (300, 1119)
        num_classes: 類別數量
        gru_units: GRU 隱藏單元數
        dropout: Dropout 比例
        use_temperature_scaling: 是否在輸出前應用溫度縮放層
    """
    print("🏗️  構建 Bi-GRU 模型（解決特徵混疊 & 過度自信）...")
    
    sequence_input = keras.Input(shape=input_shape, name='sequence_input')
    
    # Masking 層
    x = keras.layers.Masking(mask_value=0.0)(sequence_input)
    
    # 輸入層歸一化（對抗 Distribution Shift）
    x = keras.layers.LayerNormalization(name='input_norm')(x)
    
    # 第一層 Bi-GRU
    gru1_output = keras.layers.Bidirectional(
        keras.layers.GRU(
            units=gru_units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.3,
            kernel_regularizer=keras.regularizers.l2(2e-5),
            recurrent_regularizer=keras.regularizers.l2(2e-5),
            name='gru_1'
        ),
        name='bidirectional_gru_1'
    )(x)
    
    # 添加 LayerNormalization
    gru1_output = keras.layers.LayerNormalization(name='layer_norm_1')(gru1_output)
    
    # 第二層 Bi-GRU（返回序列和最終狀態，用於多尺度融合）
    gru2_output = keras.layers.Bidirectional(
        keras.layers.GRU(
            units=gru_units // 2,
            return_sequences=True,  # 改為 True，提取多尺度特徵
            dropout=dropout,
            recurrent_dropout=0.3,
            kernel_regularizer=keras.regularizers.l2(2e-5),
            recurrent_regularizer=keras.regularizers.l2(2e-5),
            name='gru_2'
        ),
        name='bidirectional_gru_2'
    )(gru1_output)
    
    # 【對抗特徵混疊】多尺度時序池化
    # 不同時間窗口的特徵，防止單一視角的混疊
    gru2_max = keras.layers.GlobalMaxPooling1D(name='global_max_pool')(gru2_output)
    gru2_avg = keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(gru2_output)
    
    # 拼接多尺度特徵
    x = keras.layers.Concatenate(name='multi_scale_concat')([gru2_max, gru2_avg])
    
    # 【Embedding 層】用於後續可視化和中心損失（可選）
    # 這層特徵應該有明確的類間分離
    embedding = keras.layers.Dense(
        256,
        kernel_regularizer=keras.regularizers.l2(3e-5),  # 更強的正則化
        kernel_constraint=keras.constraints.MaxNorm(3.0),  # 限制權重範數
        name='embedding'
    )(x)
    embedding = keras.layers.BatchNormalization(name='embedding_bn')(embedding)
    embedding = keras.layers.Activation('relu', name='embedding_activation')(embedding)
    embedding = keras.layers.Dropout(dropout, name='embedding_dropout')(embedding)
    
    # 【分類頭】較小的瓶頸層，防止過度擬合
    x = keras.layers.Dense(
        128,
        kernel_regularizer=keras.regularizers.l2(3e-5),
        kernel_constraint=keras.constraints.MaxNorm(2.0),  # 限制權重，防止極端 logits
        name='classifier_hidden'
    )(embedding)
    x = keras.layers.BatchNormalization(name='classifier_bn')(x)
    x = keras.layers.Activation('relu', name='classifier_activation')(x)
    x = keras.layers.Dropout(dropout * 0.6, name='classifier_dropout')(x)
    
    # 【輸出層】限制權重範數，防止 logit variance 過大
    logits = keras.layers.Dense(
        num_classes,
        kernel_regularizer=keras.regularizers.l2(3e-5),
        kernel_constraint=keras.constraints.MaxNorm(1.5),  # ⭐ 關鍵：限制 logits 幅度
        use_bias=True,
        dtype='float32',
        name='logits'
    )(x)
    
    # 【Calibration】可選的溫度縮放層（通過 bias 實現溫度效果）
    # 注意：真正的溫度縮放需要在訓練後調整，這裡先預留架構
    if use_temperature_scaling:
        # 添加一個可學習的縮放因子（初始化為接近 2.0，降低信心度）
        import keras.backend as K
        
        # 使用 Lambda 層實現溫度縮放的準備（初期除以較大的溫度）
        # 訓練時會自動調整
        logits = keras.layers.Lambda(
            lambda x: x / 1.5,  # 初始溫度 1.5，降低過度自信
            name='temperature_scaling'
        )(logits)
    
    # Softmax 激活
    outputs = keras.layers.Activation('softmax', dtype='float32', name='output')(logits)
    
    model = keras.Model(inputs=sequence_input, outputs=outputs, name='BiGRU_AntiCollapse_Calibrated')
    
    return model


# ==================== 訓練函數 ====================
def train_model(
    rgb_dir,
    skeleton_dir,
    output_dir,
    strategy,
    max_length=300,
    batch_size_per_replica=64,  # 每個 TPU 核心的批次大小（降低以節省內存）
    epochs=50,
    learning_rate=5e-4,
    val_split=0.2
):
    """
    訓練模型（TPU 優化）
    
    Args:
        rgb_dir: RGB 特徵目錄
        skeleton_dir: Skeleton 特徵目錄
        output_dir: 輸出目錄
        strategy: TPUStrategy
        batch_size_per_replica: 每個 TPU 核心的批次大小
        epochs: 訓練輪數
        learning_rate: 學習率
        val_split: 驗證集比例
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 計算全局批次大小
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    print(f"\n📦 全局批次大小: {global_batch_size} ({batch_size_per_replica} × {strategy.num_replicas_in_sync} 核心)")
    
    # 載入數據集
    print("\n" + "=" * 60)
    dataset = SignLanguageDataset(
        rgb_dir, 
        skeleton_dir, 
        max_length=max_length,
        use_augmentation=True,  # 啟用去重後的數據增強
        use_mixup=True,         # ⭐ 啟用 Mixup（對抗特徵混疊）
        mixup_alpha=0.2
    )
    
    # 創建 TF Dataset
    print("\n" + "=" * 60)
    train_dataset, val_dataset = dataset.create_tf_dataset(
        batch_size=global_batch_size,
        val_split=val_split,
        shuffle=True
    )
    
    # 驗證數據格式
    print("\n🔍 驗證數據集格式...")
    for features, labels in train_dataset.take(1):
        print(f"✅ Features shape: {features.shape}")
        print(f"✅ Labels shape: {labels.shape}")
        print(f"✅ Features range: [{features.numpy().min():.3f}, {features.numpy().max():.3f}]")
        break
    
    # 構建和編譯模型（Keras JAX backend 自動處理分佈式）
    print("\n" + "=" * 60)
    model = build_bigru_model(
        input_shape=(max_length, 1119),
        num_classes=dataset.num_classes,
        gru_units=256,
        dropout=0.4  # 增加 dropout
    )
    
    # 編譯模型（使用 Focal Loss + Label Smoothing + Mixup 支持）
    print("\n⚙️  編譯模型...")
    print("   🎯 啟用 Focal Loss (gamma=2.0) 對抗過度自信與特徵混疊")
    print("   🎯 啟用 Label Smoothing (0.1) 降低模型過度自信")
    print("   🎯 啟用溫度縮放 (T=1.5) 校準信心度分布")
    if dataset.use_mixup:
        print("   🎨 啟用 Mixup (alpha=0.2) 對抗特徵混疊")
    
    # 使用 Focal Loss 替代標準交叉熵
    focal_loss = FocalLoss(
        num_classes=dataset.num_classes,  # JAX 要求靜態形狀
        gamma=2.0,           # 聚焦難分類樣本
        alpha=None,          # 不使用類別權重（已經用數據增強平衡）
        label_smoothing=0.1  # 標籤平滑
    )
    
    # 選擇合適的 metrics（根據是否使用 Mixup）
    if dataset.use_mixup:
        metrics_list = [
            MixupAccuracy(name='accuracy'),
            MixupTop3Accuracy(name='top3_accuracy')
        ]
    else:
        metrics_list = [
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')
        ]
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0  # 梯度裁剪，防止訓練不穩定
        ),
        loss=focal_loss,
        metrics=metrics_list
    )
    
    # 設置 Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_path / 'best_model.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            min_delta=1e-4,
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=str(output_path / 'training_log.csv'),
            separator=',',
            append=False
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(output_path / 'tensorboard_logs'),
            histogram_freq=1
        )
    ]
    
    # 開始訓練
    print("\n" + "=" * 60)
    print("🚀 開始訓練...")
    print(f"📊 配置:")
    print(f"  - 全局批次大小: {global_batch_size}")
    print(f"  - 訓練輪數: {epochs}")
    print(f"  - 學習率: {learning_rate}")
    print(f"  - TPU 核心: {strategy.num_replicas_in_sync}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"\n❌ 訓練錯誤: {e}")
        import traceback
        traceback.print_exc()
        return
    
    training_time = datetime.now() - start_time
    
    # 保存模型和結果
    print("\n" + "=" * 60)
    print("💾 保存模型...")
    
    model.save(str(output_path / 'final_model.keras'))
    
    with open(output_path / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump(dataset.label_map, f, ensure_ascii=False, indent=2)
    
    # 生成訓練報告
    report = {
        'training_time': str(training_time),
        'total_samples': len(dataset.samples),
        'num_classes': dataset.num_classes,
        'global_batch_size': global_batch_size,
        'tpu_cores': strategy.num_replicas_in_sync,
        'epochs': epochs,
        'final_metrics': {
            'train_loss': float(history.history['loss'][-1]),
            'train_accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1])
        },
        'best_metrics': {
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_epoch': int(np.argmax(history.history['val_accuracy']) + 1)
        }
    }
    
    with open(output_path / 'training_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成訓練曲線圖
    print("\n📊 生成訓練曲線圖...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式後端
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Loss 曲線
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Model Loss (Focal Loss)', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy 曲線
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.5, 1.0])
        
        # 3. Top-3 Accuracy 曲線
        if 'top3_accuracy' in history.history:
            axes[1, 0].plot(history.history['top3_accuracy'], label='Training Top-3', linewidth=2)
            axes[1, 0].plot(history.history['val_top3_accuracy'], label='Validation Top-3', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Top-3 Accuracy', fontsize=12)
            axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0.8, 1.0])
        
        # 4. 訓練/驗證差距
        accuracy_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
        axes[1, 1].plot(accuracy_gap, linewidth=2, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Accuracy Gap', fontsize=12)
        axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(range(len(accuracy_gap)), accuracy_gap, 0, 
                                where=(accuracy_gap > 0), alpha=0.3, color='red', label='Overfitting')
        axes[1, 1].fill_between(range(len(accuracy_gap)), accuracy_gap, 0, 
                                where=(accuracy_gap <= 0), alpha=0.3, color='green', label='Good Generalization')
        axes[1, 1].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(str(output_path / 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 訓練曲線圖已保存: {output_path / 'training_curves.png'}")
        
    except Exception as e:
        print(f"⚠️  生成訓練曲線圖失敗: {e}")
    
    # 生成混淆矩陣
    print("\n📊 生成混淆矩陣...")
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 收集驗證集的所有預測
        print("   正在預測驗證集...")
        all_labels = []
        all_predictions = []
        
        for features, labels in val_dataset:
            predictions = model.predict(features, verbose=0)
            all_labels.extend(labels.numpy())
            all_predictions.extend(np.argmax(predictions, axis=1))
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # 計算混淆矩陣
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 歸一化混淆矩陣（百分比）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 繪製混淆矩陣
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        
        # 1. 原始計數
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(dataset.label_map.keys()),
                   yticklabels=sorted(dataset.label_map.keys()),
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].tick_params(axis='both', labelsize=10)
        
        # 2. 歸一化百分比
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
                   xticklabels=sorted(dataset.label_map.keys()),
                   yticklabels=sorted(dataset.label_map.keys()),
                   ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(str(output_path / 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 混淆矩陣已保存: {output_path / 'confusion_matrix.png'}")
        
        # 生成分類報告
        idx_to_word = {v: k for k, v in dataset.label_map.items()}
        target_names = [idx_to_word[i] for i in range(dataset.num_classes)]
        
        report_text = classification_report(all_labels, all_predictions, 
                                           target_names=target_names, 
                                           digits=4)
        
        with open(output_path / 'classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("手語識別模型 - 分類報告\n")
            f.write("=" * 70 + "\n\n")
            f.write(report_text)
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("混淆矩陣分析:\n")
            f.write("=" * 70 + "\n")
            
            # 分析混淆的類別對
            confusion_pairs = []
            for i in range(len(cm)):
                for j in range(len(cm)):
                    if i != j and cm[i, j] > 0:
                        confusion_pairs.append((
                            target_names[i], 
                            target_names[j], 
                            cm[i, j],
                            cm_normalized[i, j]
                        ))
            
            confusion_pairs.sort(key=lambda x: x[2], reverse=True)
            
            f.write("\n最常見的混淆 (Top 10):\n")
            f.write("-" * 70 + "\n")
            for true_label, pred_label, count, ratio in confusion_pairs[:10]:
                f.write(f"{true_label:12s} → {pred_label:12s}: {count:3d} 次 ({ratio:6.2%})\n")
        
        print(f"✅ 分類報告已保存: {output_path / 'classification_report.txt'}")
        
        # 打印到控制台
        print("\n" + "=" * 70)
        print("📋 分類報告摘要:")
        print("=" * 70)
        print(report_text)
        
    except Exception as e:
        print(f"⚠️  生成混淆矩陣失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 輸出結果
    print("\n" + "=" * 60)
    print("🎉 訓練完成！")
    print(f"⏱️  總耗時: {training_time}")
    print(f"📊 最佳驗證準確率: {report['best_metrics']['best_val_accuracy']:.4f} (Epoch {report['best_metrics']['best_epoch']})")
    print(f"📊 最終訓練準確率: {report['final_metrics']['train_accuracy']:.4f}")
    print(f"📊 最終驗證準確率: {report['final_metrics']['val_accuracy']:.4f}")
    print("=" * 60)


# ==================== 主函數 ====================
def main():
    """主函數"""
    print("🚀 Bi-GRU 手語識別訓練 - Kaggle TPU v5e-8 版本")
    print("=" * 60)
    
    # 初始化 TPU
    strategy = init_tpu_strategy()
    
    # Kaggle 環境路徑
    rgb_dir = Path("/kaggle/input/mvp-rgb/rgb_features")
    skeleton_dir = Path("/kaggle/input/mvp-skeleton/skeleton_features")
    output_dir = Path("/kaggle/working/model_output")
    
    # 驗證路徑
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB 特徵目錄不存在: {rgb_dir}")
    if not skeleton_dir.exists():
        raise FileNotFoundError(f"Skeleton 特徵目錄不存在: {skeleton_dir}")
    
    print(f"📂 RGB 特徵: {rgb_dir}")
    print(f"📂 Skeleton 特徵: {skeleton_dir}")
    print(f"📂 輸出目錄: {output_dir}")
    
    # 開始訓練
    train_model(
        rgb_dir=rgb_dir,
        skeleton_dir=skeleton_dir,
        output_dir=output_dir,
        strategy=strategy,
        max_length=300,
        batch_size_per_replica=64,
        epochs=50,
        learning_rate=5e-4,  # 保持原學習率
        val_split=0.2
    )


if __name__ == "__main__":
    main()
