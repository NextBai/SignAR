import os
import sys
import json
import hashlib
import requests
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from pathlib import Path
from werkzeug.utils import secure_filename
import time
import threading

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
socketio = SocketIO(app, cors_allowed_origins="*")

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
        return True
    except Exception as e:
        print(f"發送訊息失敗: {e}")
        return False

def send_hello_world_to_messenger():
    """發送 Hello World 到 Messenger（模擬）"""
    # 這裡可以實作實際的 Messenger 推播
    # 由於沒有特定的 recipient_id，這裡只是記錄
    print("📨 已發送 Hello World 到 Messenger Bot")
    return True

def trigger_frontend_animation(video_name="messenger_video", is_duplicate=False):
    """觸發前端動畫（用於 Messenger Bot 上傳）"""
    def run_animation():
        # 發送開始處理事件
        socketio.emit('messenger_upload', {
            'status': 'start',
            'video_name': video_name
        })
        
        # 等待動畫完成（4秒）
        time.sleep(3.5)
        
        # 發送完成事件
        message = "此影片已處理過！Hello World（重複影片）" if is_duplicate else "Hello World! 影片處理完成並已刪除"
        socketio.emit('messenger_upload', {
            'status': 'complete',
            'message': message
        })
    
    # 在背景執行緒中執行動畫
    thread = threading.Thread(target=run_animation)
    thread.daemon = True
    thread.start()

@app.route('/')
def index():
    """前端頁面"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """處理前端上傳的影片"""
    global processed_count
    
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': '沒有上傳檔案'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': '沒有選擇檔案'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': '不支援的檔案格式'}), 400
        
        # 計算影片哈希值
        video_hash = get_video_hash(file)
        
        # 檢查是否已處理過
        if video_hash in DOWNLOADED_VIDEOS:
            print(f"影片已存在，跳過儲存: {video_hash}")
            message = "此影片已處理過！Hello World（重複影片）"
        else:
            # 暫時儲存影片
            filename = secure_filename(file.filename)
            temp_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_hash}_{filename}")
            file.save(temp_path)
            
            # 記錄此影片
            DOWNLOADED_VIDEOS.add(video_hash)
            save_downloaded_videos()
            
            print(f"成功儲存影片: {temp_path}")
            
            # 模擬處理完成後刪除影片
            time.sleep(0.5)  # 短暫延遲
            try:
                os.remove(temp_path)
                print(f"已刪除影片: {temp_path}")
            except Exception as e:
                print(f"刪除影片失敗: {e}")
            
            message = "Hello World! 影片處理完成並已刪除"
        
        # 發送訊息到 Messenger
        send_hello_world_to_messenger()
        
        # 更新處理計數
        processed_count += 1
        save_processed_count()
        
        return jsonify({
            'success': True,
            'message': message,
            'video_hash': video_hash,
            'is_duplicate': video_hash in DOWNLOADED_VIDEOS
        }), 200
        
    except Exception as e:
        print(f"處理影片時發生錯誤: {e}")
        return jsonify({'success': False, 'message': f'處理失敗: {str(e)}'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """取得統計資訊"""
    return jsonify({
        'processed_count': processed_count,
        'unique_videos': len(DOWNLOADED_VIDEOS)
    }), 200

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
                                
                                # 觸發前端動畫
                                trigger_frontend_animation(f"messenger_{video_hash[:8]}", is_duplicate)
                                
                                # 檢查是否已下載過
                                if is_duplicate:
                                    print(f"⏭️ 影片已存在，跳過下載: {video_hash}")
                                    send_message(sender_id, "Hello World")
                                else:
                                    # 下載新影片
                                    print(f"⬇️ 開始下載影片...")
                                    success, file_path = download_video(video_url, video_hash)
                                    
                                    if success:
                                        DOWNLOADED_VIDEOS.add(video_hash)
                                        save_downloaded_videos()
                                        print(f"✅ 成功下載影片: {file_path}")
                                        
                                        # 處理完成後刪除影片
                                        try:
                                            os.remove(file_path)
                                            print(f"🗑️ 已刪除影片: {file_path}")
                                        except Exception as e:
                                            print(f"❌ 刪除影片失敗: {e}")
                                    else:
                                        print(f"❌ 下載影片失敗")
                                    
                                    # 無論下載成功與否，都回傳 Hello World
                                    send_message(sender_id, "Hello World")
                                
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
                    send_message(sender_id, "Hello World")
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

if __name__ == '__main__':
    print("="*60)
    print("🏭 影片處理生產線系統啟動中...")
    print("="*60)
    
    init_storage()
    
    print(f"📁 資料目錄: {DATA_DIR}")
    print(f"📄 已下載影片記錄檔: {DOWNLOADED_VIDEOS_FILE}")
    print(f"📊 處理計數檔: {PROCESSED_COUNT_FILE}")
    print(f"💾 影片儲存路徑: {VIDEO_STORAGE_PATH}")
    print(f"🔢 已處理影片數: {processed_count}")
    print(f"🎬 已記錄影片數: {len(DOWNLOADED_VIDEOS)}")
    print(f"🔑 Messenger Verify Token: {'已設定' if VERIFY_TOKEN != 'your_verify_token_here' else '⚠️ 未設定'}")
    print(f"🔐 Page Access Token: {'已設定' if PAGE_ACCESS_TOKEN != 'your_page_access_token_here' else '⚠️ 未設定'}")
    
    port = int(os.environ.get('PORT', 7860))
    print(f"🌐 啟動 WebSocket 服務於 0.0.0.0:{port}")
    print("="*60)
    print("✅ 系統就緒，等待請求...")
    print("="*60 + "\n")
    
    # 使用 SocketIO 來運行應用
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
