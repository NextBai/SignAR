# 🚫 禁用 GPU/Metal/OpenGL - 必須在所有 import 之前設定
import os

# ==================== TensorFlow GPU 禁用設定 ====================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 CUDA
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # 禁止 GPU 記憶體增長
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制所有 TF 警告（0=全部, 3=只顯示錯誤）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 優化（減少警告）

# ==================== MediaPipe GPU 禁用設定 ====================
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['MEDIAPIPE_DISABLE_EGL'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['GLOG_logtostderr'] = '1'
os.environ['GLOG_minloglevel'] = '2'  # 抑制 MediaPipe GPU 試探錯誤

import eventlet
eventlet.monkey_patch()

import sys
import json
import hashlib
import requests
from flask import Flask, request, jsonify, render_template, send_file
from flask_socketio import SocketIO, emit
from pathlib import Path
from werkzeug.utils import secure_filename
import time
import threading
import numpy as np

# 添加專案路徑到 sys.path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "feature_extraction"))
sys.path.append(str(Path(__file__).parent / "scripts"))

# 設置 Keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
# TF_CPP_MIN_LOG_LEVEL 已在上方設定為 '3'，此處不重複設定

# 強制標準輸出和錯誤輸出無緩衝，確保日誌即時顯示
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 立即輸出測試訊息
print("=" * 80, flush=True)
print("🚀 Python 應用程式開始載入...", flush=True)
print("=" * 80, flush=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'video-processing-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 上傳限制
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)

# 用於存儲已下載影片的哈希值，避免重複下載
DOWNLOADED_VIDEOS = set()
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}

# 使用可寫入的資料目錄（在 Hugging Face Spaces 使用 /app/data）
DATA_DIR = os.environ.get('DATA_DIR', '/app/data')
DOWNLOADED_VIDEOS_FILE = os.path.join(DATA_DIR, "downloaded_videos.json")
PROCESSED_COUNT_FILE = os.path.join(DATA_DIR, "processed_count.json")
VIDEO_STORAGE_PATH = os.path.join(DATA_DIR, "downloaded_videos")

# 清理配置
MAX_STORED_VIDEOS = int(os.environ.get('MAX_STORED_VIDEOS', '5'))  # 最多保留 5 個影片
MAX_VIDEO_RECORDS = int(os.environ.get('MAX_VIDEO_RECORDS', '10'))  # 最多記錄 10 筆
VIDEO_CLEANUP_DELAY = int(os.environ.get('VIDEO_CLEANUP_DELAY', '300'))  # 處理完成後 5 分鐘清理（秒）

# 處理計數器
processed_count = 0

# 從 Hugging Face Secrets 或環境變數獲取設定
VERIFY_TOKEN = os.environ.get("MESSENGER_VERIFY_TOKEN", "your_verify_token_here")
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "your_page_access_token_here")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# 手語識別器全局變量
sign_language_recognizer = None
model_loading_status = {
    'status': 'not_started',  # not_started, loading, ready, failed
    'progress': 0,
    'message': '模型尚未載入',
    'error': None
}
model_loading_lock = threading.Lock()

# ==================== 排隊系統 ====================
import queue
import asyncio
from collections import deque
from datetime import datetime, timedelta
import tempfile

# ==================== 影片預處理模組 ====================
try:
    from scripts.processor import VideoProcessor
    from utils import MediaPipePosePool, SelfieSegmentationPool
    VIDEO_PREPROCESSOR = None  # 延遲初始化
    POSE_DETECTOR_POOL = None  # MediaPipe Pose 檢測器池
    SEGMENTER_POOL = None  # SelfieSegmentation 分割器池
    PREPROCESSOR_ENABLED = True
    print("✅ scripts.processor 模組載入成功（使用 utils.mediapipe_pool）")
except ImportError as e:
    print(f"⚠️ processor.py 模組載入失敗: {e}")
    print("⚠️ 將使用原始影片處理流程（無智能裁切）")
    PREPROCESSOR_ENABLED = False

# 處理隊列
processing_queue = deque()  # [(sender_id, video_path, timestamp, language), ...]
current_processing = None  # 當前正在處理的 sender_id
queue_lock = threading.Lock()

# 用戶狀態管理
user_states = {}  # sender_id: {'status': 'waiting_language'|'queued'|'processing', 'video_path': ..., 'language': ..., 'timer': ...}
user_language_timers = {}  # sender_id: Timer對象

# 語言識別字典
LANGUAGE_KEYWORDS = {
    '英文': ['english', '英文', '英語', 'en'],
    '日文': ['japanese', '日文', '日語', '日本語', 'ja', 'jp'],
    '韓文': ['korean', '韓文', '韓語', '한국어', 'ko', 'kr'],
    '繁體中文': ['中文', '繁體中文', '繁中', 'chinese', 'zh', 'tw'],
    '簡體中文': ['简体中文', '簡體中文', '简中', 'cn'],
    '西班牙文': ['spanish', '西班牙文', '西文', 'es'],
    '法文': ['french', '法文', '法語', 'fr'],
    '德文': ['german', '德文', '德語', 'de'],
}

def detect_language(text):
    """從文字中檢測語言"""
    text_lower = text.lower().strip()
    for language, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return language
    return None

def language_selection_timeout(sender_id):
    """語言選擇超時處理（5秒後執行）"""
    with queue_lock:
        if sender_id not in user_states:
            return
        
        user_state = user_states[sender_id]
        
        # 如果用戶沒有選擇語言，預設繁體中文
        if user_state.get('language') is None:
            user_state['language'] = '繁體中文'
            print(f"⏰ 用戶 {sender_id} 語言選擇超時，預設: 繁體中文")
        
        # 更新狀態為排隊中
        user_state['status'] = 'queued'
        
        # 通知用戶排隊位置
        queue_position = get_queue_position(sender_id)
        if queue_position == 0:
            # 第一名，馬上處理
            send_message(sender_id, f"🎬 已收到您的手語影片！正在分析中，請稍候...")
        else:
            send_message(sender_id, f"📋 您目前排在第 {queue_position + 1} 位，請稍候！\n（已選擇語言: {user_state['language']}）")
        
        # 如果前3名，暫存影片（已經下載了，不需要額外處理）
        if queue_position < 3:
            print(f"💾 用戶 {sender_id} 在前3名，影片已暫存: {user_state['video_path']}")
    
    # 嘗試開始處理隊列
    process_next_in_queue()

def get_queue_position(sender_id):
    """獲取用戶在隊列中的位置（0-based）"""
    for i, (uid, _, _, _) in enumerate(processing_queue):
        if uid == sender_id:
            return i
    return -1

def add_to_queue(sender_id, video_path):
    """
    將用戶加入處理隊列
    
    Args:
        sender_id: 用戶ID
        video_path: 影片路徑
    """
    with queue_lock:
        # 檢查用戶是否已在隊列中
        for uid, _, _, _ in processing_queue:
            if uid == sender_id:
                print(f"⚠️ 用戶 {sender_id} 已在隊列中，忽略重複請求")
                send_message(sender_id, "❌ 您已經在處理隊列中，請勿重複上傳！")
                return False
        
        # 加入隊列
        processing_queue.append((sender_id, video_path, datetime.now(), None))  # language 稍後設置
        
        # 初始化用戶狀態
        user_states[sender_id] = {
            'status': 'waiting_language',
            'video_path': video_path,
            'language': None,
            'timestamp': datetime.now()
        }
        
        print(f"✅ 用戶 {sender_id} 已加入隊列（總數: {len(processing_queue)}）")

        # 先發送語言選擇提示（確保用戶先收到訊息）
        send_message(sender_id,
            "🎬 已收到您的手語影片！\n\n"
            "💬 請在 5 秒內回覆您想要的語言（例如：英文、日文、韓文）\n"
            "⏱️ 如果沒有回覆，將預設使用繁體中文。"
        )

        # 啟動 5 秒計時器等待語言選擇
        timer = threading.Timer(5.0, language_selection_timeout, args=[sender_id])
        timer.daemon = True
        timer.start()
        user_language_timers[sender_id] = timer
        
        return True

def set_user_language(sender_id, language):
    """
    設置用戶的語言偏好
    
    Args:
        sender_id: 用戶ID
        language: 語言名稱
    """
    with queue_lock:
        if sender_id not in user_states:
            return False
        
        user_state = user_states[sender_id]
        
        # 只有在 waiting_language 狀態下才能設置語言
        if user_state['status'] != 'waiting_language':
            return False
        
        user_state['language'] = language
        
        # 更新隊列中的語言
        for i, (uid, vpath, ts, _) in enumerate(processing_queue):
            if uid == sender_id:
                processing_queue[i] = (uid, vpath, ts, language)
                break
        
        print(f"✅ 用戶 {sender_id} 已設置語言: {language}")
        send_message(sender_id, f"✅ 已設定輸出語言為：{language}")
        
        return True

def process_next_in_queue():
    """處理隊列中的下一個任務"""
    global current_processing
    
    with queue_lock:
        # 如果當前有任務正在處理，不啟動新任務
        if current_processing is not None:
            print(f"⏳ 當前正在處理用戶 {current_processing}，等待完成...")
            return
        
        # 如果隊列為空，退出
        if len(processing_queue) == 0:
            print("✅ 隊列已清空")
            return
        
        # 取出第一個任務（只有狀態為 'queued' 的才處理）
        for i, (sender_id, video_path, timestamp, language) in enumerate(processing_queue):
            if sender_id in user_states and user_states[sender_id]['status'] == 'queued':
                # 移除並處理
                processing_queue.remove((sender_id, video_path, timestamp, language))
                current_processing = sender_id
                user_states[sender_id]['status'] = 'processing'
                
                print(f"🎬 開始處理用戶 {sender_id} 的影片（語言: {language or '繁體中文'}）")
                
                # 在新線程中處理影片
                thread = threading.Thread(
                    target=process_video_task,
                    args=(sender_id, video_path, language or '繁體中文'),
                    daemon=True
                )
                thread.start()
                break

def preprocess_video_async(video_path):
    """
    異步預處理影片（智能裁切 + 標準化）

    使用 MediaPipePosePool 和 SelfieSegmentationPool 進行處理

    Args:
        video_path: 原始影片路徑

    Returns:
        preprocessed_path: 預處理後的影片路徑（臨時文件）
        success: 是否成功
    """
    global VIDEO_PREPROCESSOR, POSE_DETECTOR_POOL, SEGMENTER_POOL

    if not PREPROCESSOR_ENABLED:
        # 如果預處理器未啟用，返回原始影片
        return video_path, True

    try:
        # 延遲初始化 VideoProcessor（避免啟動時開銷）
        if VIDEO_PREPROCESSOR is None:
            print("🔧 初始化 VideoProcessor（首次使用）...")

            # 直接使用 MediaPipePosePool 初始化檢測器
            try:
                pose_detector = MediaPipePosePool.get_detector(min_detection_confidence=0.5)
                POSE_DETECTOR_POOL = pose_detector
                print("  ✅ MediaPipe Pose 檢測器已初始化")
            except Exception as e:
                print(f"  ⚠️ MediaPipe Pose 初始化失敗: {e}")

            # 直接使用 SelfieSegmentationPool 初始化分割器
            try:
                segmenter = SelfieSegmentationPool.get_segmenter(model_selection=1)
                SEGMENTER_POOL = segmenter
                print("  ✅ SelfieSegmentation 分割器已初始化")
            except Exception as e:
                print(f"  ⚠️ SelfieSegmentation 初始化失敗: {e}")

            # ⚡ 辨識模式：不設定 target_frames，保持原始幀數
            # 原因：
            # 1. 訓練時固定 80 幀是為了批次訓練（模型要求固定輸入）
            # 2. 辨識時使用滑動窗口（SlidingWindowInference, stride=80）
            # 3. 可處理任意長度影片（例如包含多個單詞的長句子）
            # 4. 如果設定 target_frames=80，會把長影片壓縮，導致識別錯誤
            VIDEO_PREPROCESSOR = VideoProcessor(
                enable_cropping=True,
                enable_background_removal=False,  # 根據需求調整
                target_frames=None  # ✅ 明確設為 None，不限制幀數
            )
            print("✅ VideoProcessor 初始化完成（辨識模式：無幀數限制）")

        # 創建臨時文件存儲預處理後的影片
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        preprocessed_path = temp_file.name
        temp_file.close()

        print(f"📹 開始預處理影片: {Path(video_path).name}")
        print(f"  - 智能裁切: 啟用（使用 MediaPipePosePool）")
        print(f"  - 背景移除: 啟用（使用 SelfieSegmentationPool）")
        print(f"  - 標準化: 80幀 @ 30fps, 224x224")

        # 執行預處理（augmentor=None，僅標準化）
        success = VIDEO_PREPROCESSOR.process_video(
            input_path=video_path,
            output_path=preprocessed_path,
            augmentor=None  # 辨識時不使用數據增強
        )

        if success:
            print(f"✅ 預處理完成: {Path(preprocessed_path).name}")
            return preprocessed_path, True
        else:
            print(f"⚠️ 預處理失敗，使用原始影片")
            # 清理失敗的臨時文件
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
            return video_path, False

    except Exception as e:
        print(f"❌ 預處理異常: {e}")
        import traceback
        traceback.print_exc()
        return video_path, False

def process_video_task(sender_id, video_path, target_language):
    """
    處理影片任務（在獨立線程中執行）
    
    完整流程：
    1. 影片預處理（智能裁切 + 標準化）
    2. 手語識別（滑動窗口 + OpenAI）
    3. 語言轉換
    4. 清理臨時文件
    
    Args:
        sender_id: 用戶ID
        video_path: 原始影片路徑
        target_language: 目標語言
    """
    global current_processing, processed_count
    
    preprocessed_path = None
    
    try:
        print(f"🎬 處理用戶 {sender_id} 的影片: {video_path}")
        print(f"🌐 目標語言: {target_language}")
        
        # ==================== 步驟 1: 影片預處理 ====================
        # 發送預處理開始事件
        with app.app_context():
            socketio.emit('processing_progress', {
                'progress': 0,
                'current': 0,
                'total': 100,
                'message': '開始處理影片 - 預處理階段',
                'timestamp': time.time()
            }, namespace='/')
        
        preprocessed_path, preprocess_success = preprocess_video_async(video_path)
        
        if preprocess_success and preprocessed_path != video_path:
            print(f"✅ 使用預處理後的影片進行識別")
            video_to_process = preprocessed_path
        else:
            print(f"⚠️ 使用原始影片進行識別（預處理失敗或未啟用）")
            video_to_process = video_path
        
        # 發送預處理完成事件
        with app.app_context():
            socketio.emit('processing_progress', {
                'progress': 20,
                'current': 20,
                'total': 100,
                'message': '預處理完成 - 開始特徵提取',
                'timestamp': time.time()
            }, namespace='/')
        
        # ==================== 步驟 2: 手語識別 ====================
        recognized_sentence = process_video_and_get_sentence_with_language(
            video_to_process, 
            socketio, 
            target_language
        )
        
        # ==================== 步驟 3: 發送結果 ====================
        send_message(sender_id, f"✅ 識別完成！\n\n📝 結果：{recognized_sentence}")
        
        # 更新計數
        processed_count += 1
        save_processed_count()
        
        print(f"✅ 用戶 {sender_id} 處理完成")
        
    except Exception as e:
        print(f"❌ 處理用戶 {sender_id} 的影片時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        send_message(sender_id, "❌ 抱歉，影片處理失敗，請稍後再試。")
    
    finally:
        # ==================== 步驟 4: 清理臨時文件 ====================
        if preprocessed_path and preprocessed_path != video_path:
            try:
                if os.path.exists(preprocessed_path):
                    os.remove(preprocessed_path)
                    print(f"🧹 已清理預處理臨時文件: {Path(preprocessed_path).name}")
            except Exception as e:
                print(f"⚠️ 清理臨時文件失敗: {e}")

        # ==================== 步驟 5: 清理 MediaPipe 資源 ====================
        try:
            # 清理當前線程的 MediaPipe 實例（使用 pool 的清理方法）
            MediaPipePosePool.close_thread_detector()
            SelfieSegmentationPool.close_thread_segmenter()
            print("🧹 已清理 MediaPipe 線程資源")
        except Exception as e:
            print(f"⚠️ 清理 MediaPipe 資源失敗: {e}")

        # 清理用戶狀態
        with queue_lock:
            if sender_id in user_states:
                user_states[sender_id]['status'] = 'completed'
                # 延遲刪除狀態（防止立即重複）
                threading.Timer(10.0, lambda: user_states.pop(sender_id, None)).start()

            if sender_id in user_language_timers:
                user_language_timers.pop(sender_id, None)

            current_processing = None
            print(f"🔓 處理完成，釋放處理鎖")

        # 處理下一個任務
        process_next_in_queue()

# 初始化
def init_storage():
    """初始化儲存目錄和已下載影片記錄"""
    global processed_count
    
    Path(VIDEO_STORAGE_PATH).mkdir(exist_ok=True)
    
    if os.path.exists(DOWNLOADED_VIDEOS_FILE):
        with open(DOWNLOADED_VIDEOS_FILE, 'r') as f:
            try:
                data = json.load(f)
                DOWNLOADED_VIDEOS.update(data)
            except json.JSONDecodeError:
                pass
    
    # 載入處理計數
    if os.path.exists(PROCESSED_COUNT_FILE):
        with open(PROCESSED_COUNT_FILE, 'r') as f:
            try:
                data = json.load(f)
                processed_count = data.get('count', 0)
            except json.JSONDecodeError:
                processed_count = 0

def load_model_async():
    """異步載入模型（背景執行）"""
    global sign_language_recognizer, model_loading_status
    
    with model_loading_lock:
        if model_loading_status['status'] == 'loading':
            return  # 已經在載入中，避免重複
        model_loading_status['status'] = 'loading'
        model_loading_status['message'] = '開始載入模型...'
        model_loading_status['progress'] = 0
    
    try:
        from sliding_window_inference import SlidingWindowInference
        
        model_path = Path(__file__).parent / 'model_output' / 'best_model_mps.keras'
        label_path = Path(__file__).parent / 'model_output' / 'label_map.json'
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        if not label_path.exists():
            raise FileNotFoundError(f"標籤文件不存在: {label_path}")
        
        print("🔧 背景載入手語識別器...")
        
        # 更新進度
        with model_loading_lock:
            model_loading_status['message'] = '載入 Keras 模型...'
            model_loading_status['progress'] = 20
        
        sign_language_recognizer = SlidingWindowInference(
            model_path=str(model_path),
            label_map_path=str(label_path),
            device='cpu',
            stride=80,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 完成
        with model_loading_lock:
            model_loading_status['status'] = 'ready'
            model_loading_status['message'] = '模型載入完成'
            model_loading_status['progress'] = 100
        
        print("✅ 手語識別器初始化成功（背景載入）")
        
    except Exception as e:
        error_msg = f"模型載入失敗: {e}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        with model_loading_lock:
            model_loading_status['status'] = 'failed'
            model_loading_status['message'] = error_msg
            model_loading_status['error'] = str(e)

def get_sign_language_recognizer():
    """獲取手語識別器（檢查載入狀態）"""
    global sign_language_recognizer
    
    # 如果還沒開始載入，啟動背景載入
    if model_loading_status['status'] == 'not_started':
        thread = threading.Thread(target=load_model_async, daemon=True)
        thread.start()
    
    return sign_language_recognizer

def save_downloaded_videos():
    """儲存已下載影片的記錄"""
    with open(DOWNLOADED_VIDEOS_FILE, 'w') as f:
        json.dump(list(DOWNLOADED_VIDEOS), f)

def save_processed_count():
    """儲存處理計數"""
    with open(PROCESSED_COUNT_FILE, 'w') as f:
        json.dump({'count': processed_count}, f)

def cleanup_old_videos():
    """清理舊影片檔案，保留最新的 N 個"""
    try:
        if not os.path.exists(VIDEO_STORAGE_PATH):
            return

        # 獲取所有影片檔案及其修改時間
        video_files = []
        for filename in os.listdir(VIDEO_STORAGE_PATH):
            if filename.endswith('.mp4'):
                filepath = os.path.join(VIDEO_STORAGE_PATH, filename)
                mtime = os.path.getmtime(filepath)
                video_hash = filename[:-4]  # 移除 .mp4
                video_files.append((video_hash, filepath, mtime))

        # 按時間排序（最新的在前）
        video_files.sort(key=lambda x: x[2], reverse=True)

        # 保留最新的 MAX_STORED_VIDEOS 個，刪除其他
        files_to_delete = video_files[MAX_STORED_VIDEOS:]

        for video_hash, filepath, _ in files_to_delete:
            try:
                os.remove(filepath)
                print(f"🗑️  已刪除舊影片: {video_hash}")

                # 從記錄中移除
                if video_hash in DOWNLOADED_VIDEOS:
                    DOWNLOADED_VIDEOS.discard(video_hash)
            except Exception as e:
                print(f"⚠️  刪除影片失敗 {video_hash}: {e}")

        if files_to_delete:
            save_downloaded_videos()
            print(f"✅ 清理完成，保留 {len(video_files) - len(files_to_delete)} 個影片")

    except Exception as e:
        print(f"❌ 清理影片失敗: {e}")
        import traceback
        traceback.print_exc()

def cleanup_old_records():
    """清理舊記錄，限制 DOWNLOADED_VIDEOS 大小"""
    global DOWNLOADED_VIDEOS

    try:
        if len(DOWNLOADED_VIDEOS) > MAX_VIDEO_RECORDS:
            # 獲取所有記錄及其對應檔案的時間
            records_with_time = []
            for video_hash in DOWNLOADED_VIDEOS:
                filepath = os.path.join(VIDEO_STORAGE_PATH, f"{video_hash}.mp4")
                if os.path.exists(filepath):
                    mtime = os.path.getmtime(filepath)
                    records_with_time.append((video_hash, mtime))
                else:
                    # 檔案不存在，標記為最舊
                    records_with_time.append((video_hash, 0))

            # 按時間排序，保留最新的
            records_with_time.sort(key=lambda x: x[1], reverse=True)
            new_records = set([hash for hash, _ in records_with_time[:MAX_VIDEO_RECORDS]])

            removed_count = len(DOWNLOADED_VIDEOS) - len(new_records)
            DOWNLOADED_VIDEOS = new_records
            save_downloaded_videos()

            print(f"🗑️  清理舊記錄：移除 {removed_count} 筆，保留 {len(DOWNLOADED_VIDEOS)} 筆")

    except Exception as e:
        print(f"❌ 清理記錄失敗: {e}")

def schedule_video_cleanup(video_hash, delay=VIDEO_CLEANUP_DELAY):
    """延遲清理特定影片（給前端播放時間）"""
    def delayed_cleanup():
        try:
            filepath = os.path.join(VIDEO_STORAGE_PATH, f"{video_hash}.mp4")
            if os.path.exists(filepath):
                # 檢查是否還在保留範圍內
                video_files = []
                for filename in os.listdir(VIDEO_STORAGE_PATH):
                    if filename.endswith('.mp4'):
                        fp = os.path.join(VIDEO_STORAGE_PATH, filename)
                        video_files.append((fp, os.path.getmtime(fp)))

                video_files.sort(key=lambda x: x[1], reverse=True)

                # 如果不在保留範圍內，刪除
                if (filepath, os.path.getmtime(filepath)) not in video_files[:MAX_STORED_VIDEOS]:
                    os.remove(filepath)
                    print(f"🗑️  延遲清理：已刪除 {video_hash}")

                    if video_hash in DOWNLOADED_VIDEOS:
                        DOWNLOADED_VIDEOS.discard(video_hash)
                        save_downloaded_videos()
        except Exception as e:
            print(f"⚠️  延遲清理失敗 {video_hash}: {e}")

    # 延遲執行清理
    timer = threading.Timer(delay, delayed_cleanup)
    timer.daemon = True
    timer.start()

def cleanup_memory():
    """強制 Python 垃圾回收，釋放記憶體"""
    import gc
    gc.collect()
    print("🧹 記憶體清理完成")

def allowed_file(filename):
    """檢查檔案副檔名是否允許"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_hash(video_url_or_file):
    """計算影片 URL 或檔案的哈希值"""
    if isinstance(video_url_or_file, str):
        # URL 的情況
        return hashlib.md5(video_url_or_file.encode()).hexdigest()
    else:
        # 檔案物件的情況，計算檔案內容的哈希
        video_url_or_file.seek(0)
        file_hash = hashlib.md5()
        chunk_size = 8192
        while chunk := video_url_or_file.read(chunk_size):
            file_hash.update(chunk)
        video_url_or_file.seek(0)
        return file_hash.hexdigest()

def download_video(video_url, video_hash):
    """下載影片並儲存"""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        file_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_hash}.mp4")
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True, file_path
    except Exception as e:
        print(f"下載影片失敗: {e}")
        return False, None

def send_message(recipient_id, message_text):
    """發送訊息給使用者"""
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"✅ 成功發送訊息給 {recipient_id}")
        return True
    except Exception as e:
        print(f"❌ 發送訊息失敗: {e}")
        return False

def process_video_and_get_sentence_with_language(video_path, socketio_instance=None, target_language='繁體中文'):
    """處理影片並返回指定語言的識別句子"""
    try:
        print(f"🎬 開始處理影片: {video_path}")
        print(f"🌐 目標語言: {target_language}")
        
        # 獲取全局識別器（檢查載入狀態）
        recognizer = get_sign_language_recognizer()
        
        # 檢查模型載入狀態
        if model_loading_status['status'] == 'loading':
            message = f"模型載入中...({model_loading_status['progress']}%)"
            print(f"⏳ {message}")
            return message
        elif model_loading_status['status'] == 'failed':
            print(f"❌ 模型載入失敗: {model_loading_status['error']}")
            return f"模型載入失敗: {model_loading_status['message']}"
        elif recognizer is None:
            print("⚠️ 手語識別器未就緒")
            return "手語識別器未就緒，請稍後再試"
        
        # 創建進度回調函數
        def progress_callback(current, total, message):
            if socketio_instance:
                progress_percent = int((current / total) * 100) if total > 0 else 0
                socketio_instance.emit('processing_progress', {
                    'progress': progress_percent,
                    'current': current,
                    'total': total,
                    'message': message,
                    'timestamp': time.time()
                }, namespace='/')
                print(f"📊 進度: {progress_percent}% - {message}")
        
        # 設定進度回調
        recognizer.progress_callback = progress_callback
        
        # 處理影片（不保存 JSON 結果）
        results = recognizer.process_video(
            video_path=video_path,
            save_results=False
        )
        
        # 使用 OpenAI 重組句子（帶語言轉換）
        if OPENAI_API_KEY and recognizer.openai_client:
            sentence, explanation = recognizer.compose_sentence_with_openai(results, target_language=target_language)
            print(f"✅ 識別完成 ({target_language}): {sentence}")
            
            # 發送完成事件
            if socketio_instance:
                video_hash = None
                if video_path and isinstance(video_path, str):
                    filename = os.path.basename(video_path)
                    if filename.endswith('.mp4'):
                        video_hash = filename[:-4]
                
                send_processing_complete(video_hash, sentence)
            
            return sentence
        else:
            # 如果沒有 OpenAI，返回 Top-1 單詞序列
            words = [result['top5'][0]['word'] for result in results]
            sentence = ' '.join(words)
            print(f"✅ 識別完成 (無 OpenAI): {sentence}")
            
            # 發送完成事件
            if socketio_instance:
                video_hash = None
                if video_path and isinstance(video_path, str):
                    filename = os.path.basename(video_path)
                    if filename.endswith('.mp4'):
                        video_hash = filename[:-4]
                
                send_processing_complete(video_hash, sentence)
            
            return sentence

    except Exception as e:
        print(f"❌ 影片處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return "Hello World! (處理失敗)"

    finally:
        # 無論成功或失敗，都清理記憶體
        try:
            import gc
            gc.collect()
            print("🧹 處理完成後記憶體清理")
        except:
            pass

def process_video_and_get_sentence(video_path, socketio_instance=None):
    """處理影片並返回識別的句子（預設繁體中文）"""
    return process_video_and_get_sentence_with_language(video_path, socketio_instance, '繁體中文')

def process_video_and_get_sentence_legacy(video_path, socketio_instance=None):
    """舊版處理函數（保留向後兼容）"""
    try:
        print(f"🎬 開始處理影片: {video_path}")
        
        # 獲取全局識別器（檢查載入狀態）
        recognizer = get_sign_language_recognizer()
        
        # 檢查模型載入狀態
        if model_loading_status['status'] == 'loading':
            message = f"模型載入中...({model_loading_status['progress']}%)"
            print(f"⏳ {message}")
            return message
        elif model_loading_status['status'] == 'failed':
            print(f"❌ 模型載入失敗: {model_loading_status['error']}")
            return f"模型載入失敗: {model_loading_status['message']}"
        elif recognizer is None:
            print("⚠️ 手語識別器未就緒")
            return "手語識別器未就緒，請稍後再試"
        
        # 創建進度回調函數
        def progress_callback(current, total, message):
            if socketio_instance:
                progress_percent = int((current / total) * 100) if total > 0 else 0
                socketio_instance.emit('processing_progress', {
                    'progress': progress_percent,
                    'current': current,
                    'total': total,
                    'message': message,
                    'timestamp': time.time()
                }, namespace='/')
                print(f"📊 進度: {progress_percent}% - {message}")
        
        # 設定進度回調
        recognizer.progress_callback = progress_callback
        
        # 處理影片（不保存 JSON 結果）
        results = recognizer.process_video(
            video_path=video_path,
            save_results=False
        )
        
        # 使用 OpenAI 重組句子
        if OPENAI_API_KEY and recognizer.openai_client:
            sentence, explanation = recognizer.compose_sentence_with_openai(results)
            print(f"✅ 識別完成: {sentence}")
            
            # 發送完成事件
            if socketio_instance:
                video_hash = None
                if video_path and isinstance(video_path, str):
                    filename = os.path.basename(video_path)
                    if filename.endswith('.mp4'):
                        video_hash = filename[:-4]
                
                send_processing_complete(video_hash, sentence)
            
            return sentence
        else:
            # 如果沒有 OpenAI，返回 Top-1 單詞序列
            words = [result['top5'][0]['word'] for result in results]
            sentence = ' '.join(words)
            print(f"✅ 識別完成 (無 OpenAI): {sentence}")
            
            # 發送完成事件
            if socketio_instance:
                video_hash = None
                if video_path and isinstance(video_path, str):
                    filename = os.path.basename(video_path)
                    if filename.endswith('.mp4'):
                        video_hash = filename[:-4]
                
                send_processing_complete(video_hash, sentence)
            
            return sentence
            
    except Exception as e:
        print(f"❌ 影片處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return "Hello World! (處理失敗)"

def send_processing_complete(video_hash, recognized_sentence):
    """發送處理完成事件"""
    with app.app_context():
        socketio.emit('messenger_upload', {
            'status': 'complete',
            'message': f"識別結果: {recognized_sentence}",
            'recognized_sentence': recognized_sentence,
            'video_url': f'/videos/{video_hash}' if video_hash else None,
            'video_hash': video_hash,
            'timestamp': time.time()
        }, namespace='/')
        print(f"🔔 已發送完成動畫事件: {recognized_sentence}")

        # 處理完成後執行清理
        try:
            # 1. 清理舊影片檔案（保留最新 5 個）
            cleanup_old_videos()

            # 2. 清理舊記錄（限制最多 10 筆）
            cleanup_old_records()

            # 3. 延遲清理當前影片（5 分鐘後，給前端播放時間）
            if video_hash:
                schedule_video_cleanup(video_hash)

            # 4. 強制記憶體清理
            cleanup_memory()

        except Exception as e:
            print(f"⚠️  清理過程出錯: {e}")

def trigger_frontend_animation(video_name="messenger_video", video_hash=None, is_duplicate=False, recognized_sentence=None):
    """觸發前端動畫（用於 Messenger Bot 上傳）"""
    def run_animation():
        with app.app_context():
            # 發送開始處理事件（包含影片 URL，讓前端立即顯示影片）
            socketio.emit('messenger_upload', {
                'status': 'start',
                'video_name': video_name,
                'video_url': f'/videos/{video_hash}' if video_hash else None,
                'video_hash': video_hash,
                'timestamp': time.time(),
                'recognized_sentence': recognized_sentence or "處理中..."
            }, namespace='/')

            print(f"🔔 已發送開始動畫事件: {video_name} (video_hash: {video_hash})")

            # 不等待固定時間，而是等待處理完成訊號
            # 進度會通過 processing_progress 事件即時更新

            # 等待完成事件（這個會由 process_video_and_get_sentence 完成後觸發）
            # 這裡我們不手動發送完成事件，而是讓它自然結束

    # 在背景執行緒中執行動畫
    thread = threading.Thread(target=run_animation)
    thread.daemon = True
    thread.start()

@app.route('/')
def index():
    """前端頁面"""
    return render_template('index.html')

@app.route('/stats', methods=['GET'])
def get_stats():
    """取得統計資訊"""
    return jsonify({
        'processed_count': processed_count,
        'unique_videos': len(DOWNLOADED_VIDEOS)
    }), 200

@app.route('/videos/<video_hash>')
def serve_video(video_hash):
    """提供影片檔案（支援範圍請求，實現即時串流播放）"""
    file_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_hash}.mp4")
    if os.path.exists(file_path):
        return send_file(
            file_path,
            mimetype='video/mp4',
            conditional=True,  # 啟用範圍請求支援（Range Requests）
            max_age=3600  # 緩存 1 小時
        )
    else:
        return jsonify({'error': 'Video not found'}), 404

@app.route('/api/videos', methods=['GET'])
def get_videos():
    """取得所有影片清單"""
    videos = []
    for hash_val in DOWNLOADED_VIDEOS:
        file_path = os.path.join(VIDEO_STORAGE_PATH, f"{hash_val}.mp4")
        if os.path.exists(file_path):
            videos.append({
                'hash': hash_val,
                'url': f'/videos/{hash_val}',
                'timestamp': os.path.getmtime(file_path),
                'size': os.path.getsize(file_path)
            })
    # 依時間排序（最新的在前）
    videos.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(videos), 200

@app.route('/webhook', methods=['GET'])
def verify():
    """Webhook 驗證端點"""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    if mode and token:
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            print('WEBHOOK_VERIFIED')
            return challenge, 200
        else:
            return 'Forbidden', 403
    
    return 'Webhook endpoint', 200

@app.route('/webhook', methods=['POST'])
def webhook():
    """處理 Messenger 的 Webhook 事件（支援排隊系統）"""
    global processed_count
    data = request.get_json()
    
    # 添加詳細日誌
    print(f"📥 收到 Webhook 請求")
    print(f"📋 請求資料: {json.dumps(data, indent=2, ensure_ascii=False)}")
    
    if data.get('object') == 'page':
        print(f"✅ 確認為 Page 物件")
        for entry in data.get('entry', []):
            print(f"📨 處理 Entry: {entry.get('id')}")
            for messaging_event in entry.get('messaging', []):
                sender_id = messaging_event['sender']['id']
                print(f"👤 發送者 ID: {sender_id}")
                
                # 處理影片訊息
                if messaging_event.get('message', {}).get('attachments'):
                    attachments = messaging_event['message']['attachments']
                    print(f"📎 找到 {len(attachments)} 個附件")
                    
                    for attachment in attachments:
                        attachment_type = attachment.get('type')
                        print(f"📄 附件類型: {attachment_type}")
                        
                        if attachment_type == 'video':
                            video_url = attachment.get('payload', {}).get('url')
                            print(f"🎬 影片 URL: {video_url}")
                            
                            if video_url:
                                video_hash = get_video_hash(video_url)
                                is_duplicate = video_hash in DOWNLOADED_VIDEOS
                                
                                print(f"🔑 影片哈希: {video_hash}")
                                print(f"🔄 是否重複: {is_duplicate}")

                                # 下載影片（無論是否重複，都需要本地文件）
                                if is_duplicate:
                                    print(f"⏭️ 影片已存在，使用現有檔案: {video_hash}")
                                    file_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_hash}.mp4")
                                else:
                                    # 下載新影片
                                    print(f"⬇️ 開始下載影片...")
                                    success, file_path = download_video(video_url, video_hash)

                                    if success:
                                        DOWNLOADED_VIDEOS.add(video_hash)
                                        save_downloaded_videos()
                                        print(f"✅ 成功下載影片: {file_path}")
                                        print(f"💾 影片已保留供前端播放")

                                        # 下載後立即檢查並清理（確保不超過限制）
                                        try:
                                            cleanup_old_videos()
                                            cleanup_old_records()
                                        except Exception as e:
                                            print(f"⚠️  下載後清理失敗: {e}")
                                    else:
                                        print(f"❌ 下載影片失敗")
                                        send_message(sender_id, "❌ 抱歉，影片下載失敗，請稍後再試。")
                                        continue
                                
                                # 加入排隊系統
                                success = add_to_queue(sender_id, file_path)
                                
                                if success:
                                    # 觸發前端動畫
                                    trigger_frontend_animation(
                                        video_name=f"messenger_{video_hash[:8]}",
                                        video_hash=video_hash,
                                        is_duplicate=is_duplicate,
                                        recognized_sentence="排隊處理中..."
                                    )
                        else:
                            print(f"⚠️ 非影片附件，類型為: {attachment_type}")
                
                # 處理文字訊息
                elif messaging_event.get('message', {}).get('text'):
                    message_text = messaging_event['message']['text']
                    print(f"💬 收到文字訊息: {message_text}")
                    
                    # 檢查是否是語言選擇
                    detected_language = detect_language(message_text)
                    
                    if sender_id in user_states:
                        user_state = user_states[sender_id]
                        
                        # 如果用戶在等待語言選擇狀態
                        if user_state['status'] == 'waiting_language':
                            if detected_language:
                                # 設置語言
                                set_user_language(sender_id, detected_language)
                            else:
                                # 無法識別的語言，提示用戶
                                send_message(sender_id, 
                                    "❓ 無法識別語言，請輸入明確的語言名稱\n"
                                    "（例如：英文、日文、韓文、繁體中文等）"
                                )
                        elif user_state['status'] == 'queued':
                            # 已經在排隊中，友善提示
                            queue_position = get_queue_position(sender_id)
                            send_message(sender_id, 
                                f"✅ 您的影片正在排隊處理中（第 {queue_position + 1} 位）\n"
                                f"語言已設定為：{user_state.get('language', '繁體中文')}\n"
                                f"請耐心等候，我們會盡快完成！"
                            )
                        elif user_state['status'] == 'processing':
                            # 正在處理中
                            send_message(sender_id, 
                                f"🎬 您的影片正在處理中，請稍候...\n"
                                f"輸出語言：{user_state.get('language', '繁體中文')}"
                            )
                        else:
                            # 其他狀態，提示上傳影片
                            send_message(sender_id, "🎬 請傳送手語影片給我，我會幫您識別內容！")
                    else:
                        # 用戶不在任何狀態中，提示上傳影片
                        send_message(sender_id, "🎬 請傳送手語影片給我，我會幫您識別內容！")
                
                else:
                    print(f"⚠️ 未知的訊息類型: {messaging_event}")
    else:
        print(f"❌ 不是 Page 物件: {data.get('object')}")
    
    print(f"✅ Webhook 處理完成\n")
    return 'OK', 200

@app.route('/queue_status', methods=['GET'])
def queue_status():
    """查詢排隊系統狀態"""
    with queue_lock:
        queue_info = []
        for i, (sender_id, video_path, timestamp, language) in enumerate(processing_queue):
            queue_info.append({
                'position': i + 1,
                'sender_id': sender_id[:8] + '...',  # 隱私保護
                'language': language or '繁體中文',
                'timestamp': timestamp.isoformat(),
                'status': user_states.get(sender_id, {}).get('status', 'unknown')
            })
        
        return jsonify({
            'queue_length': len(processing_queue),
            'current_processing': current_processing[:8] + '...' if current_processing else None,
            'queue': queue_info
        }), 200

@app.route('/health', methods=['GET'])
def health():
    """健康檢查端點"""
    with queue_lock:
        queue_length = len(processing_queue)
        is_processing = current_processing is not None
    
    return jsonify({
        "status": "healthy",
        "model_status": model_loading_status['status'],
        "model_progress": model_loading_status['progress'],
        "model_message": model_loading_status['message'],
        "downloaded_videos_count": len(DOWNLOADED_VIDEOS),
        "processed_count": processed_count,
        "queue_length": queue_length,
        "is_processing": is_processing,
        "data_dir": DATA_DIR,
        "verify_token_set": VERIFY_TOKEN != "your_verify_token_here",
        "page_token_set": PAGE_ACCESS_TOKEN != "your_page_access_token_here"
    }), 200

@app.route('/debug', methods=['GET'])
def debug():
    """調試端點 - 顯示系統詳細資訊"""
    return jsonify({
        "system_info": {
            "data_dir": DATA_DIR,
            "downloaded_videos_file": DOWNLOADED_VIDEOS_FILE,
            "processed_count_file": PROCESSED_COUNT_FILE,
            "video_storage_path": VIDEO_STORAGE_PATH,
        },
        "statistics": {
            "processed_count": processed_count,
            "unique_videos": len(DOWNLOADED_VIDEOS),
            "downloaded_video_hashes": list(DOWNLOADED_VIDEOS)[:5] if len(DOWNLOADED_VIDEOS) > 0 else []
        },
        "configuration": {
            "verify_token_set": VERIFY_TOKEN != "your_verify_token_here",
            "page_token_set": PAGE_ACCESS_TOKEN != "your_page_access_token_here",
            "max_content_length": app.config['MAX_CONTENT_LENGTH'],
            "allowed_extensions": list(ALLOWED_EXTENSIONS)
        },
        "directories": {
            "data_dir_exists": os.path.exists(DATA_DIR),
            "video_storage_exists": os.path.exists(VIDEO_STORAGE_PATH),
            "downloaded_videos_file_exists": os.path.exists(DOWNLOADED_VIDEOS_FILE),
            "processed_count_file_exists": os.path.exists(PROCESSED_COUNT_FILE)
        }
    }), 200

@app.route('/test-websocket', methods=['GET'])
def test_websocket():
    """測試 WebSocket 廣播"""
    socketio.emit('messenger_upload', {
        'status': 'start',
        'video_name': 'test_video'
    }, namespace='/')

    time.sleep(1)

    socketio.emit('messenger_upload', {
        'status': 'complete',
        'message': 'WebSocket 測試成功！',
        'video_url': None,
        'video_hash': 'test123',
        'timestamp': time.time()
    }, namespace='/')

    return jsonify({'message': 'WebSocket 事件已發送，檢查前端 Console'}), 200


def cleanup_resources():
    """應用關閉時清理所有資源"""
    print("\n🧹 應用關閉中，清理資源...")
    try:
        # 清理所有 MediaPipe 實例（使用 pool 的清理方法）
        MediaPipePosePool.close_all()
        SelfieSegmentationPool.close_all()
        print("✅ 已清理所有 MediaPipe 資源")
    except Exception as e:
        print(f"⚠️ 清理 MediaPipe 資源時出錯: {e}")

    print("👋 應用已關閉\n")


# 註冊應用關閉處理器
import atexit
atexit.register(cleanup_resources)

if __name__ == '__main__':
    print("="*60)
    print("🏭 手語影片識別系統啟動中...")
    print("="*60)
    
    # 初始化儲存
    init_storage()
    
    print(f"📁 資料目錄: {DATA_DIR}")
    print(f"📄 已下載影片記錄檔: {DOWNLOADED_VIDEOS_FILE}")
    print(f"📊 處理計數檔: {PROCESSED_COUNT_FILE}")
    print(f"💾 影片儲存路徑: {VIDEO_STORAGE_PATH}")
    print(f"🔢 已處理影片數: {processed_count}")
    print(f"🎬 已記錄影片數: {len(DOWNLOADED_VIDEOS)}")
    print(f"🔑 Messenger Verify Token: {'✅ 已設定' if VERIFY_TOKEN != 'your_verify_token_here' else '⚠️ 未設定'}")
    print(f"🔐 Page Access Token: {'✅ 已設定' if PAGE_ACCESS_TOKEN != 'your_page_access_token_here' else '⚠️ 未設定'}")
    print(f"🔐 OpenAI API Key: {'✅ 已設定' if OPENAI_API_KEY else '⚠️ 未設定'}")
    
    port = int(os.environ.get('PORT', 7860))
    print(f"🌐 啟動 WebSocket 服務於 0.0.0.0:{port}")
    print(f"🔧 使用 async_mode: eventlet")
    print("="*60)
    print("✅ 系統就緒！")
    print("🚀 啟動背景異步載入模型...")
    print("="*60 + "\n")
    
    # 🚀 啟動背景異步載入模型（不阻塞 Flask 啟動）
    model_thread = threading.Thread(target=load_model_async, daemon=True)
    model_thread.start()

    try:
        # 使用 SocketIO 來運行應用（eventlet 模式）
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\n⏹️ 收到中斷信號...")
    finally:
        # 應用關閉時清理資源
        cleanup_resources()
