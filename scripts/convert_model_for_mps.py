#!/usr/bin/env python3
"""
模型轉換腳本：Kaggle TPU (JAX backend + bfloat16) → M1 MPS (TensorFlow backend + float32)

使用方式：
1. 從 Kaggle 下載 best_model.keras 和 label_map.json
2. 執行此腳本轉換模型
3. 轉換後的模型可在 M1 MPS 上使用
"""

import os
import sys
from pathlib import Path

# 設置環境變數（在導入 Keras 之前）
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
import json
import numpy as np


# ==================== 自定義層定義（必須在載入模型前定義）====================
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
        """簡化版本，僅用於載入"""
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
        config.update({
            'num_classes': self.num_classes,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing
        })
        return config


@keras.saving.register_keras_serializable()
class MixupAccuracy(keras.metrics.Metric):
    """Mixup Accuracy - 用於載入訓練好的模型"""
    
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
    """Mixup Top-3 Accuracy - 用於載入訓練好的模型"""
    
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


def convert_model(input_model_path, output_model_path):
    """
    轉換 Keras 模型：JAX backend + bfloat16 → TensorFlow backend + float32
    
    Args:
        input_model_path: 輸入模型路徑（來自 Kaggle）
        output_model_path: 輸出模型路徑（M1 MPS 兼容）
    """
    print("=" * 70)
    print("🔄 Keras 模型轉換：Kaggle TPU → M1 MPS")
    print("=" * 70)
    
    input_path = Path(input_model_path)
    output_path = Path(output_model_path)
    
    if not input_path.exists():
        print(f"❌ 輸入模型不存在: {input_path}")
        return False
    
    # 確保輸出目錄存在
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📥 載入原始模型: {input_path}")
    print(f"   Backend: Keras 3 (自動檢測)")
    
    try:
        # 1. 載入模型（Keras 3 會自動處理不同 backend）
        # 啟用不安全反序列化以處理 Lambda 層（Temperature Scaling）
        print("\n🔧 載入模型...")
        print("   ⚠️  啟用 unsafe_deserialization（處理 Lambda 層）")
        keras.config.enable_unsafe_deserialization()
        
        # 使用 custom_objects 載入自定義層
        custom_objects = {
            'FocalLoss': FocalLoss,
            'MixupAccuracy': MixupAccuracy,
            'MixupTop3Accuracy': MixupTop3Accuracy
        }
        
        model = keras.models.load_model(input_path, custom_objects=custom_objects)
        
        print(f"✅ 模型載入成功")
        print(f"   輸入形狀: {model.input_shape}")
        print(f"   輸出形狀: {model.output_shape}")
        print(f"   參數量: {model.count_params():,}")
        
        # 2. 檢查並設置精度策略
        print("\n🔧 轉換精度策略...")
        current_policy = keras.mixed_precision.global_policy()
        print(f"   當前策略: {current_policy}")
        
        # 設置為 float32（M1 MPS 兼容）
        keras.mixed_precision.set_global_policy('float32')
        print(f"   新策略: float32")
        
        # 3. 重新構建模型以應用新精度
        print("\n🔧 重新構建模型...")
        
        # 獲取模型配置
        config = model.get_config()
        
        # 重新構建模型
        new_model = keras.Model.from_config(config)
        
        # 複製權重
        print("🔧 複製權重...")
        for old_layer, new_layer in zip(model.layers, new_model.layers):
            try:
                weights = old_layer.get_weights()
                if weights:
                    # 轉換為 float32
                    weights_float32 = [w.astype('float32') for w in weights]
                    new_layer.set_weights(weights_float32)
            except Exception as e:
                print(f"⚠️  層 {new_layer.name} 權重複製失敗: {e}")
        
        print("✅ 權重複製完成")
        
        # 4. 編譯模型（使用與訓練時相同的配置）
        print("\n🔧 編譯模型...")
        print("   使用 Focal Loss + Mixup Metrics（與訓練時一致）")
        
        new_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
            loss=FocalLoss(num_classes=15, gamma=2.0, label_smoothing=0.1),
            metrics=[
                MixupAccuracy(name='accuracy'),
                MixupTop3Accuracy(name='top3_accuracy')
            ]
        )
        print("✅ 模型編譯完成")
        
        # 5. 保存轉換後的模型
        print(f"\n💾 保存轉換後的模型: {output_path}")
        new_model.save(output_path)
        print("✅ 模型保存成功")
        
        # 6. 驗證轉換
        print("\n🔍 驗證轉換...")
        verify_model = keras.models.load_model(output_path)
        
        # 測試推論
        import numpy as np
        dummy_input = np.random.randn(1, 300, 1119).astype('float32')
        
        with tf.device('/CPU:0'):  # 先在 CPU 上測試
            output = verify_model.predict(dummy_input, verbose=0)
        
        print(f"✅ 驗證成功！輸出形狀: {output.shape}")
        
        # 7. 輸出統計
        original_size = input_path.stat().st_size / (1024 * 1024)
        converted_size = output_path.stat().st_size / (1024 * 1024)
        
        print("\n" + "=" * 70)
        print("📊 轉換統計：")
        print("=" * 70)
        print(f"原始模型大小: {original_size:.2f} MB")
        print(f"轉換後大小: {converted_size:.2f} MB")
        print(f"大小差異: {(converted_size - original_size):.2f} MB ({(converted_size/original_size - 1)*100:+.1f}%)")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 轉換失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_mps_compatibility(model_path):
    """驗證模型在 M1 MPS 上的兼容性"""
    print("\n🔍 驗證 M1 MPS 兼容性...")
    
    try:
        # 檢查 TensorFlow Metal
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ 檢測到 {len(gpus)} 個 GPU 設備（Metal）")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            
            # 載入模型並在 GPU 上測試
            model = keras.models.load_model(model_path)
            
            import numpy as np
            dummy_input = np.random.randn(2, 300, 1119).astype('float32')
            
            print("\n🧪 在 MPS 上測試推論...")
            with tf.device('/GPU:0'):
                output = model.predict(dummy_input, verbose=0)
            
            print(f"✅ MPS 推論成功！輸出形狀: {output.shape}")
            print("🎉 模型完全兼容 M1 MPS！")
            
            return True
        else:
            print("⚠️  未檢測到 Metal GPU，模型將在 CPU 上運行")
            return False
            
    except Exception as e:
        print(f"❌ MPS 兼容性測試失敗: {e}")
        return False


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='轉換 Keras 模型為 M1 MPS 兼容格式')
    parser.add_argument('--input', type=str, 
                       default='model_output/best_model.keras',
                       help='輸入模型路徑（來自 Kaggle）')
    parser.add_argument('--output', type=str,
                       default='model_output/best_model_mps.keras',
                       help='輸出模型路徑（M1 MPS 兼容）')
    parser.add_argument('--verify', action='store_true',
                       help='驗證 MPS 兼容性')
    
    args = parser.parse_args()
    
    # 轉換模型
    success = convert_model(args.input, args.output)
    
    if success and args.verify:
        # 驗證 MPS 兼容性
        verify_mps_compatibility(args.output)
    
    if success:
        print("\n✅ 轉換完成！現在可以使用 inference.py 進行推論")
        print(f"   模型路徑: {args.output}")
    else:
        print("\n❌ 轉換失敗，請檢查錯誤信息")
        sys.exit(1)


if __name__ == "__main__":
    main()

