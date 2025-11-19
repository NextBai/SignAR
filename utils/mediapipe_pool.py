#!/usr/bin/env python3
"""
MediaPipe 線程池統一管理
支援 Pose、Holistic、SelfieSegmentation
線程安全，自動資源管理
"""

import threading
try:
    import mediapipe as mp
except ImportError:
    raise ImportError("請安裝 mediapipe: pip install mediapipe")


class MediaPipePosePool:
    """
    MediaPipe Pose 線程池管理
    - 每個線程使用獨立的檢測器實例（避免時間戳記衝突）
    - 使用 ThreadLocal 確保線程隔離
    - 自動資源管理
    """
    _thread_local = threading.local()
    _lock = threading.Lock()
    _all_instances = []

    @classmethod
    def get_detector(cls, min_detection_confidence=0.5, model_complexity=1):
        """
        獲取當前線程專用的 Pose 檢測器實例

        Args:
            min_detection_confidence: 最低檢測信心度
            model_complexity: 模型複雜度 (0=快速, 1=均衡, 2=精確)

        Returns:
            MediaPipe Pose 檢測器實例（線程專用）
        """
        # 檢查當前線程是否已有實例，並驗證實例是否有效
        needs_new_detector = False

        if not hasattr(cls._thread_local, 'detector') or cls._thread_local.detector is None:
            needs_new_detector = True
        else:
            # 檢查實例的 _graph 是否有效（避免 _graph is None 錯誤）
            try:
                if not hasattr(cls._thread_local.detector, '_graph') or cls._thread_local.detector._graph is None:
                    needs_new_detector = True
            except:
                needs_new_detector = True

        if needs_new_detector:
            with cls._lock:
                mp_pose = mp.solutions.pose
                detector = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.5
                )
                cls._thread_local.detector = detector
                cls._all_instances.append(detector)

        return cls._thread_local.detector

    @classmethod
    def close_thread_detector(cls):
        """清理當前線程的檢測器"""
        if hasattr(cls._thread_local, 'detector') and cls._thread_local.detector is not None:
            cls._thread_local.detector.close()
            cls._thread_local.detector = None

    @classmethod
    def close_all(cls):
        """清理所有線程的檢測器"""
        with cls._lock:
            for detector in cls._all_instances:
                try:
                    detector.close()
                except:
                    pass
            cls._all_instances.clear()


class MediaPipeHolisticPool:
    """
    MediaPipe Holistic 線程池管理
    - 為 Jupyter ThreadPoolExecutor 環境提供線程隔離
    - 每個線程使用獨立的 Holistic 實例，避免時間戳記衝突
    """
    _thread_local = threading.local()
    _lock = threading.Lock()
    _all_instances = []

    @classmethod
    def get_holistic(cls, model_complexity=0, refine_face_landmarks=True):
        """
        獲取當前線程專用的 Holistic 實例

        Args:
            model_complexity: 模型複雜度 (0=快速, 1=均衡, 2=精確)
            refine_face_landmarks: 是否啟用精細面部特徵點（用於瞳孔檢測）

        Returns:
            MediaPipe Holistic 實例（線程專用）
        """
        # 檢查當前線程是否已有實例，並驗證實例是否有效
        needs_new_holistic = False

        if not hasattr(cls._thread_local, 'holistic') or cls._thread_local.holistic is None:
            needs_new_holistic = True
        else:
            # 檢查實例的 _graph 是否有效（避免 _graph is None 錯誤）
            try:
                if not hasattr(cls._thread_local.holistic, '_graph') or cls._thread_local.holistic._graph is None:
                    needs_new_holistic = True
            except:
                needs_new_holistic = True

        if needs_new_holistic:
            with cls._lock:
                mp_holistic = mp.solutions.holistic
                holistic = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    smooth_segmentation=False,
                    refine_face_landmarks=refine_face_landmarks,
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


class SelfieSegmentationPool:
    """
    MediaPipe SelfieSegmentation 線程池管理
    - 每個線程使用獨立的分割器實例（避免時間戳記衝突）
    - 使用 ThreadLocal 確保線程隔離
    """
    _thread_local = threading.local()
    _lock = threading.Lock()
    _all_instances = []

    @classmethod
    def get_segmenter(cls, model_selection=1):
        """
        獲取當前線程專用的 SelfieSegmentation 實例

        Args:
            model_selection: 模型選擇 (0=一般品質/速度, 1=較高品質/較慢速度)

        Returns:
            MediaPipe SelfieSegmentation 實例（線程專用）
        """
        # 檢查當前線程是否已有實例，並驗證實例是否有效
        needs_new_segmenter = False

        if not hasattr(cls._thread_local, 'segmenter') or cls._thread_local.segmenter is None:
            needs_new_segmenter = True
        else:
            # 檢查實例的 _graph 是否有效（避免 _graph is None 錯誤）
            try:
                if not hasattr(cls._thread_local.segmenter, '_graph') or cls._thread_local.segmenter._graph is None:
                    needs_new_segmenter = True
            except:
                needs_new_segmenter = True

        if needs_new_segmenter:
            with cls._lock:
                mp_selfie_segmentation = mp.solutions.selfie_segmentation
                segmenter = mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=model_selection
                )
                cls._thread_local.segmenter = segmenter
                cls._all_instances.append(segmenter)

        return cls._thread_local.segmenter

    @classmethod
    def close_thread_segmenter(cls):
        """清理當前線程的分割器"""
        if hasattr(cls._thread_local, 'segmenter') and cls._thread_local.segmenter is not None:
            cls._thread_local.segmenter.close()
            cls._thread_local.segmenter = None

    @classmethod
    def close_all(cls):
        """清理所有線程的分割器"""
        with cls._lock:
            for segmenter in cls._all_instances:
                try:
                    segmenter.close()
                except:
                    pass
            cls._all_instances.clear()


def _is_jupyter_environment():
    """檢測是否在 Jupyter/IPython 環境中運行"""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False
