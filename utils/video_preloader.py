#!/usr/bin/env python3
"""
統一的異步視頻預載入器
支援 RGB 和灰階兩種模式
減少 I/O 等待時間，提升處理效率
"""

import cv2
import queue
import threading
from pathlib import Path


class VideoPreloader:
    """
    異步影片預載入器 - 提前讀取影片幀到內存

    特性：
    - 支援 RGB 和灰階兩種模式
    - 可設定最大幀數限制
    - 線程安全的隊列管理
    - 自動錯誤處理
    """

    def __init__(self, max_queue_size=4, color_mode='rgb', max_frames=300):
        """
        初始化預載入器

        Args:
            max_queue_size: 預載入隊列最大大小
            color_mode: 顏色模式 ('rgb' 或 'gray')
            max_frames: 每個視頻最多讀取的幀數
        """
        self.video_queue = queue.Queue()  # 待處理的影片任務 (video_path, output_dir)
        self.frame_queue = queue.Queue(maxsize=max_queue_size)  # 預載入的幀數據
        self.stop_event = threading.Event()
        self.preload_thread = None
        self.color_mode = color_mode
        self.max_frames = max_frames

    def start(self):
        """啟動預載入線程"""
        self.preload_thread = threading.Thread(target=self._preload_worker, daemon=True)
        self.preload_thread.start()

    def stop(self):
        """停止預載入線程"""
        self.stop_event.set()
        if self.preload_thread:
            self.preload_thread.join(timeout=5)

    def add_video(self, video_path, output_dir=None):
        """
        添加影片到預載入隊列

        Args:
            video_path: 影片路徑
            output_dir: 輸出目錄（可選）
        """
        self.video_queue.put((video_path, output_dir))

    def get_preloaded_frames(self, timeout=1):
        """
        獲取預載入的幀數據

        Args:
            timeout: 超時時間（秒）

        Returns:
            (video_path, frames, output_dir, frame_width, frame_height) 或 None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _preload_worker(self):
        """預載入工作線程"""
        while not self.stop_event.is_set():
            try:
                # 獲取任務
                task = self.video_queue.get(timeout=0.1)
                if task is None:  # 哨兵值
                    self.frame_queue.put(None)
                    break

                video_path, output_dir = task

                # 讀取影片幀
                frames, frame_width, frame_height = self._load_video_frames(video_path)

                # 放入預載入隊列
                try:
                    self.frame_queue.put(
                        (video_path, frames, output_dir, frame_width, frame_height),
                        timeout=1
                    )
                except queue.Full:
                    # 如果隊列滿，跳過這個影片
                    continue

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 預載入失敗 {video_path}: {e}")

    def _load_video_frames(self, video_path):
        """
        讀取影片幀

        Args:
            video_path: 影片路徑

        Returns:
            (frames, frame_width, frame_height) 或 (None, 0, 0)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None, 0, 0

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames = []
            frame_count = 0

            while frame_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # 根據顏色模式處理
                if self.color_mode == 'gray':
                    # 轉換為灰階
                    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif self.color_mode == 'rgb':
                    # 轉換為 RGB
                    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # 保持 BGR
                    frame_processed = frame

                frames.append(frame_processed)
                frame_count += 1

            cap.release()
            return frames, frame_width, frame_height

        except Exception as e:
            print(f"❌ 讀取影片失敗 {video_path}: {e}")
            return None, 0, 0


class AsyncBatchPreloader:
    """
    異步批次預載入器 - 專為 skeleton 提取器設計
    在計算時同時讀取下一個視頻，減少 I/O 等待時間
    """

    def __init__(self, video_queue, output_queue, max_queue_size=3):
        """
        初始化批次預載入器

        Args:
            video_queue: 輸入任務隊列
            output_queue: 輸出結果隊列
            max_queue_size: 最大隊列大小
        """
        self.video_queue = video_queue
        self.output_queue = output_queue
        self.max_queue_size = max_queue_size
        self.stop_event = threading.Event()

    def run(self):
        """背景預載入線程"""
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

                    # 讀取所有幀
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
        """停止預載入線程"""
        self.stop_event.set()
