# 🚫 禁用 GPU/Metal/OpenGL - 必須在所有 import 之前設定
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['MEDIAPIPE_DISABLE_EGL'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['GLOG_logtostderr'] = '1'
# 抑制 MediaPipe GPU 試探的錯誤訊息（2=只顯示 ERROR 以上）
os.environ['GLOG_minloglevel'] = '2'

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

# 設置 Keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def process_video_and_get_sentence(video_path, socketio_instance=None):
    """處理影片並返回識別的句子"""
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

def trigger_frontend_animation(video_name="messenger_video", video_hash=None, is_duplicate=False, recognized_sentence=None):
    """觸發前端動畫（用於 Messenger Bot 上傳）"""
    def run_animation():
        with app.app_context():
            # 發送開始處理事件
            socketio.emit('messenger_upload', {
                'status': 'start',
                'video_name': video_name,
                'recognized_sentence': recognized_sentence or "處理中..."
            }, namespace='/')

            print(f"🔔 已發送開始動畫事件: {video_name}")

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
    """提供影片檔案"""
    file_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_hash}.mp4")
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='video/mp4')
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
    """處理 Messenger 的 Webhook 事件"""
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

                                # 檢查是否已下載過
                                if is_duplicate:
                                    print(f"⏭️ 影片已存在，跳過下載: {video_hash}")
                                    file_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_hash}.mp4")
                                    
                                    # 處理影片並獲取識別結果
                                    recognized_sentence = process_video_and_get_sentence(file_path, socketio)
                                    
                                    # 觸發前端動畫（重複影片）
                                    trigger_frontend_animation(
                                        video_name=f"messenger_{video_hash[:8]}",
                                        video_hash=video_hash,
                                        is_duplicate=True,
                                        recognized_sentence=recognized_sentence
                                    )
                                    
                                    # 發送識別結果給用戶
                                    send_message(sender_id, recognized_sentence)
                                else:
                                    # 下載新影片
                                    print(f"⬇️ 開始下載影片...")
                                    success, file_path = download_video(video_url, video_hash)

                                    if success:
                                        DOWNLOADED_VIDEOS.add(video_hash)
                                        save_downloaded_videos()
                                        print(f"✅ 成功下載影片: {file_path}")
                                        print(f"💾 影片已保留供前端播放")

                                        # 處理影片並獲取識別結果
                                        recognized_sentence = process_video_and_get_sentence(file_path, socketio)
                                        
                                        # 觸發前端動畫（新影片）
                                        trigger_frontend_animation(
                                            video_name=f"messenger_{video_hash[:8]}",
                                            video_hash=video_hash,
                                            is_duplicate=False,
                                            recognized_sentence=recognized_sentence
                                        )
                                        
                                        # 發送識別結果給用戶
                                        send_message(sender_id, recognized_sentence)
                                    else:
                                        print(f"❌ 下載影片失敗")
                                        send_message(sender_id, "抱歉，影片下載失敗")
                                
                                # 更新處理計數
                                processed_count += 1
                                save_processed_count()
                                print(f"📊 處理計數已更新: {processed_count}")
                        else:
                            print(f"⚠️ 非影片附件，類型為: {attachment_type}")
                
                # 處理一般文字訊息
                elif messaging_event.get('message', {}).get('text'):
                    message_text = messaging_event['message']['text']
                    print(f"💬 收到文字訊息: {message_text}")
                    send_message(sender_id, "請傳送手語影片給我，我會幫您識別內容！")
                else:
                    print(f"⚠️ 未知的訊息類型: {messaging_event}")
    else:
        print(f"❌ 不是 Page 物件: {data.get('object')}")
    
    print(f"✅ Webhook 處理完成\n")
    return 'OK', 200

@app.route('/health', methods=['GET'])
def health():
    """健康檢查端點"""
    return jsonify({
        "status": "healthy",
        "model_status": model_loading_status['status'],
        "model_progress": model_loading_status['progress'],
        "model_message": model_loading_status['message'],
        "downloaded_videos_count": len(DOWNLOADED_VIDEOS),
        "processed_count": processed_count,
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

    # 使用 SocketIO 來運行應用（eventlet 模式）
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
