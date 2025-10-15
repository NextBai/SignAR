# 手語影片識別系統 🏭🤟

基於 Flask + WebSocket 的手語影片識別系統，整合 Facebook Messenger Bot 和深度學習模型，提供即時手語識別與翻譯服務。

## 功能特點

- 🤟 **手語識別**：使用 BiGRU + 注意力機制識別手語動作
- ✂️ **智能裁切**：MediaPipe Pose 自動聚焦簽名者，去除背景干擾
- 🎯 **自適應處理**：自動插值/重採樣，支援任意長度影片
- 🌐 **多語言翻譯**：支援繁中、英文、日文、韓文等語言輸出
- 📊 **排隊系統**：一次處理一個影片，避免用戶衝突
- 🎨 **生產線視覺介面**：動畫化的影片處理流程
- 📹 **Messenger Bot 整合**：接收 Facebook Messenger 傳來的影片
- 🎬 **即時影片播放**：前端自動載入並播放收到的影片
- 🤖 **AI 句子重組**：使用 OpenAI 將識別結果組合成完整句子
- 💾 **智慧重複檢測**：MD5 哈希值避免重複處理
- 💬 **WebSocket 即時同步**：Messenger Bot 觸發前端動畫
- 📊 **統計追蹤**：即時處理數量統計

## 系統架構

### 技術堆疊
- **前端**: HTML/CSS/JavaScript + Socket.IO
- **後端**: Flask + Flask-SocketIO + Eventlet
- **深度學習**: 
  - TensorFlow/Keras (模型推論)
  - PyTorch (特徵提取)
  - MediaPipe (骨架檢測)
  - OpenCV (影片處理)
- **AI 服務**: OpenAI GPT (句子重組)
- **部署**: Render / Docker
- **影片儲存**: 本地檔案系統（需持久化儲存）

### 識別流程

#### **完整處理管線**

```
用戶上傳影片 → Messenger Bot 接收
    ↓
【步驟 1】影片預處理（智能優化）
    ├─ MediaPipe Pose 人體檢測
    ├─ 智能裁切：聚焦簽名者上半身 + 手臂
    ├─ 邊界框擴展（15% padding）
    ├─ 幀數標準化（至少 80 幀）
    ├─ 解析度標準化（224×224）
    └─ 儲存至臨時文件
    ↓
【步驟 2】影片載入與正規化
    ├─ 讀取預處理後的影片
    ├─ 幀數調整策略：
    │   • < 80幀 且 差距 ≤ 20幀 → 線性插值補齊
    │   • < 80幀 且 差距 > 20幀 → 均勻重複填充
    │   • ≥ 80幀 → 保持原始長度或標準重採樣
    └─ 確保最終影片 ≥ 80 幀
    ↓
【步驟 3】滑動窗口特徵提取
    ├─ 窗口大小：80 幀（~2.67秒 @ 30fps）
    ├─ 滑動步長：80 幀（無重疊，可調整）
    ├─ 每個窗口並行提取：
    │   • RGB 特徵（ResNet-50）→ 300×512
    │   • 骨架特徵（MediaPipe Holistic）→ 300×607
    │   • 合併特徵矩陣：300×1119
    └─ 窗口數量 = (總幀數 - 80) ÷ 步長 + 1
    ↓
【步驟 4】BiGRU 模型推論
    ├─ 輸入：300×1119 特徵矩陣
    ├─ BiGRU + 多頭注意力機制
    ├─ 輸出：每個窗口的 Top-5 預測
    └─ 信心度：70-85%（經校準）
    ↓
【步驟 5】OpenAI 句子重組 + 語言翻譯
    ├─ 從每個窗口 Top-5 選擇最佳單詞
    ├─ 去除相鄰窗口的重複單詞
    ├─ 組合成流暢句子
    ├─ 翻譯至目標語言（繁中/英/日/韓）
    └─ 附帶解釋（選擇邏輯）
    ↓
【步驟 6】結果回傳 + 清理
    ├─ 透過 Messenger 發送識別結果
    ├─ 刪除臨時預處理文件
    ├─ 更新處理統計
    └─ 處理下一個排隊任務
```

#### **關鍵技術亮點**

1. **智能裁切（MediaPipe Pose）**
   - 自動檢測並聚焦簽名者
   - 去除背景干擾
   - 確保手臂完整性
   - 15% padding 避免裁切不完整

2. **自適應幀數處理**
   - 影片太短 → 自動插值/填充至 80 幀
   - 影片過長 → 滑動窗口掃描
   - FPS 不一致 → 標準重採樣至 30fps

3. **並行特徵提取**
   - RGB 和骨架特徵同時提取
   - 多線程加速處理
   - 降低延遲至 2-3 秒/窗口

4. **AI 智能重組**
   - 非貪婪選擇（不一定選 Top-1）
   - 考慮前後文脈
   - 多語言翻譯支援

## 使用方式

### Messenger Bot

#### **完整使用流程**

1. **上傳影片**
   - 發送手語影片到 Facebook Messenger Bot
   - Bot 立即回覆：「✅ 影片接收成功！請在 5 秒內選擇目標語言...」

2. **選擇語言（5 秒內，可選）**
   - 發送語言關鍵字：「English」、「日文」、「Korean」等
   - 未發送 → 預設使用「繁體中文」
   - Bot 回覆：「🌐 已設定目標語言：XXX」

3. **排隊等待**
   - 如果有其他影片正在處理，Bot 回覆：
     「⏳ 您目前排在第 X 位，請稍候...」
   - 如果您是第一位，立即開始處理

4. **影片處理**
   - 前端即時顯示生產線動畫
   - 處理階段：
     1. 智能裁切（MediaPipe Pose）
     2. 滑動窗口掃描
     3. 特徵提取（RGB + Skeleton）
     4. BiGRU 模型推論
     5. OpenAI 句子重組 + 翻譯

5. **接收結果**
   - Bot 自動回覆識別的句子
   - 範例：
     - 繁體中文：「我愛你」
     - 英文：「I love you」
     - 日文：「愛しています」
     - 韓文：「사랑해요」

#### **排隊機制**

- **一次處理一個影片**：避免伺服器過載
- **FIFO 順序**：先上傳先處理
- **用戶輪替**：同一用戶處理完後排到最後
- **友善提示**：Spam 或無關訊息會收到提醒

### 手語單詞支援
系統目前支援 15 個常用手語單詞：
- hello, thank you, please, sorry, love
- yes, no, help, student, teacher
- family, friend, name, how, what

## 環境變數設定

需要設定以下環境變數：

```bash
# Messenger Bot 設定（必需）
MESSENGER_VERIFY_TOKEN=你的驗證令牌
PAGE_ACCESS_TOKEN=你的頁面存取令牌

# OpenAI 設定（選填，用於句子重組）
OPENAI_API_KEY=你的 OpenAI API 金鑰

# 系統設定（選填）
SECRET_KEY=flask_secret_key
DATA_DIR=/app/data          # 影片儲存目錄
PORT=10000                  # 服務端口

# TensorFlow 設定（自動配置）
KERAS_BACKEND=tensorflow
TF_CPP_MIN_LOG_LEVEL=2
PYTHONUNBUFFERED=1
```

## 部署到 Zeabur

### 前置準備
1. 確保專案包含以下檔案：
   - `app.py` - 主應用程式
   - `sliding_window_inference.py` - 手語識別核心
   - `requirements.txt` - Python 依賴
   - `Dockerfile` - Docker 配置
   - `model_output/` - 模型文件目錄
     - `best_model_mps.keras` - 訓練好的模型
     - `label_map.json` - 標籤映射
   - `feature_extraction/` - 特徵提取模組
   - `templates/` - 前端模板

### 1. 建立 Service
1. 登入 [Zeabur](https://zeabur.com/)
2. 點擊 "Create Project" → "New Service"
3. 連接你的 GitHub 倉庫
4. 選擇此專案

### 2. 設定部署配置
- **Name**: `signar`（或自訂）
- **Environment**: `Docker`（推薦）
- **Branch**: `main`

#### 使用 Docker 部署（推薦）
- **Build Command**: 自動偵測 Dockerfile
- **Start Command**: 自動使用 Dockerfile 的 CMD
- **Port**: `10000`

### 3. 設定環境變數
在 Zeabur Dashboard > Environment Variables 中新增：

```bash
# 必需變數
MESSENGER_VERIFY_TOKEN=你的驗證令牌
PAGE_ACCESS_TOKEN=你的頁面存取令牌

# 選填變數
OPENAI_API_KEY=你的 OpenAI API 金鑰
DATA_DIR=/app/data
PORT=10000

# 系統變數（自動配置）
KERAS_BACKEND=tensorflow
TF_CPP_MIN_LOG_LEVEL=2
PYTHONUNBUFFERED=1
```

### 4. 啟用持久化儲存（重要！）
手語影片需要保存在持久化儲存中：

- 在 Zeabur Dashboard > Storage 新增 Volume
- **Name**: `video-storage`
- **Mount Path**: `/app/data`
- **Size**: 建議至少 2GB（每個影片約 5-50MB）

### 5. 資源配置建議
由於深度學習模型需要較多資源：

- **Instance Type**: 建議選擇有至少 2GB RAM 的方案
- **CPU**: 建議至少 1-2 核心（推論速度更快）
- **Storage**: 至少 5GB（包含模型文件 + 影片）

### 6. 部署
點擊 "Deploy"，Zeabur 會自動：
1. Clone 倉庫
2. 安裝依賴（可能需要 10-15 分鐘，包含 TensorFlow、PyTorch、MediaPipe）
3. 載入模型
4. 啟動服務

### 7. 驗證部署
部署完成後，訪問以下端點確認：

- `https://signar.zeabur.app/` - 前端介面（生產線視覺化）
- `https://signar.zeabur.app/health` - 健康檢查
- `https://signar.zeabur.app/debug` - 系統資訊

健康檢查應返回：
```json
{
  "status": "healthy",
  "downloaded_videos_count": 0,
  "processed_count": 0,
  "verify_token_set": true,
  "page_token_set": true
}
```

#### 生產環境狀態
目前系統已成功部署在：**https://signar.zeabur.app/**

## Messenger Webhook 設定

### 設定步驟
1. 前往 [Facebook Developers Console](https://developers.facebook.com/)
2. 選擇你的應用 → Messenger → 設定
3. 設定 Webhook URL：`https://signar.zeabur.app/webhook`
4. 設定 Verify Token 為你的 `MESSENGER_VERIFY_TOKEN`
5. 訂閱以下事件：
   - ✅ `messages`
   - ✅ `messaging_postbacks`
6. 選擇 Facebook 粉絲專頁並訂閱

### 測試 Webhook
```bash
# 測試驗證端點
curl "https://signar.zeabur.app/webhook?hub.mode=subscribe&hub.verify_token=YOUR_TOKEN&hub.challenge=test123"

# 應該返回：test123
```

### 生產環境測試
```bash
# 檢查健康狀態
curl "https://signar.zeabur.app/health"

# 檢查系統狀態
curl "https://signar.zeabur.app/debug"
```

### 常見問題

#### Q: Webhook 驗證失敗？
- 確認 `MESSENGER_VERIFY_TOKEN` 與 Facebook 設定一致
- 檢查 Zeabur 服務是否正常運行：`https://signar.zeabur.app/health`
- 查看 Zeabur Logs 確認收到驗證請求

#### Q: Bot 沒有回應？
- 確認 `PAGE_ACCESS_TOKEN` 正確且未過期
- 檢查粉絲專頁是否正確訂閱
- 查看 Zeabur Logs 確認收到訊息

## Docker 部署（可選）

如果使用 Docker 部署到其他平台：

```bash
# 建立映像
docker build -t sign-language-recognition .

# 運行容器
docker run -d \
  -p 10000:10000 \
  -e MESSENGER_VERIFY_TOKEN=你的令牌 \
  -e PAGE_ACCESS_TOKEN=你的令牌 \
  -e OPENAI_API_KEY=你的金鑰 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_output:/app/model_output \
  sign-language-recognition
```

### Docker Compose（推薦）

建立 `docker-compose.yml`：

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "10000:10000"
    environment:
      - MESSENGER_VERIFY_TOKEN=${MESSENGER_VERIFY_TOKEN}
      - PAGE_ACCESS_TOKEN=${PAGE_ACCESS_TOKEN}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATA_DIR=/app/data
      - PORT=10000
    volumes:
      - ./data:/app/data
      - ./model_output:/app/model_output
    restart: unless-stopped
```

啟動：
```bash
docker-compose up -d
```

## API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 前端介面 |
| `/webhook` | GET | Webhook 驗證 |
| `/webhook` | POST | 處理 Messenger 訊息 |
| `/videos/<hash>` | GET | 取得影片檔案 |
| `/api/videos` | GET | 取得影片清單（JSON）|
| `/queue_status` | GET | 查詢處理隊列狀態 |
| `/stats` | GET | 統計資訊 |
| `/health` | GET | 健康檢查（包含隊列長度）|
| `/debug` | GET | 調試資訊（包含模型狀態）|

## 專案結構

```
MVP拷貝2/
├── app.py                          # 主應用程式（Flask + 手語識別整合）
├── sliding_window_inference.py     # 滑動窗口手語識別核心
├── requirements.txt                # Python 依賴
├── Dockerfile                      # Docker 配置
├── README.md                       # 專案文檔
│
├── model_output/                   # 模型文件（需包含在部署中）
│   ├── best_model_mps.keras       # 訓練好的模型
│   └── label_map.json             # 標籤映射（15 個手語單詞）
│
├── feature_extraction/             # 特徵提取模組
│   ├── rgb_feature_extraction.py  # RGB 特徵（ResNet-50）
│   └── skeleton_feature_extraction.py  # 骨架特徵（MediaPipe）
│
├── templates/                      # 前端模板
│   └── index.html                 # 主頁面
│
├── config/                         # 配置文件
│   ├── requirements_inference.txt
│   └── .env
│
├── data/                          # 數據目錄（需持久化）
│   └── downloaded_videos/         # 儲存影片
│
├── scripts/                       # 其他腳本
│   ├── processor.py              # 影片預處理（智能裁切 + 標準化）
│   ├── inference.py              # 獨立推論腳本
│   └── realtime_inference.py     # 即時推論（網路攝影機）
│
└── training/                      # 訓練相關
    └── train_bigrunet_tpu.py     # 模型訓練腳本
```

## 語言支援

### **輸入語言**
- 手語：台灣手語（TSL）為主

### **輸出語言**
用戶可在上傳影片後 5 秒內指定目標語言：
- 🇹🇼 **繁體中文**（預設）
- 🇺🇸 **英文** - 發送 "English" 或 "英文"
- 🇯🇵 **日文** - 發送 "Japanese" 或 "日文"
- 🇰🇷 **韓文** - 發送 "Korean" 或 "韓文"

### **使用範例**
```
用戶：[上傳手語影片]
Bot：✅ 影片接收成功！請在 5 秒內選擇目標語言...

用戶：English
Bot：🌐 已設定目標語言：English
     ⏳ 您目前排在第 1 位，請稍候...

[處理中...]

Bot：✅ 識別完成！
     📝 結果：I love you
```

## 影片處理規格

### **輸入要求**
- **格式**: MP4, AVI, MOV, MKV, FLV, WMV, WebM
- **長度**: 任意長度（系統自動處理）
  - < 80 幀（~2.67秒）→ 自動插值/填充
  - ≥ 80 幀 → 滑動窗口掃描
- **建議拍攝**:
  - 正面拍攝，簽名者在畫面中
  - 完整上半身 + 雙手可見
  - 充足光線，避免背光

### **預處理流程**

#### **1. 智能裁切（VideoProcessor）**
- **技術**: MediaPipe Pose 人體檢測
- **目標**: 自動聚焦簽名者
- **處理**:
  - 檢測關鍵點：鼻子、眼睛、肩膀、手肘、手腕、手指
  - 計算邊界框（包含完整上半身 + 手臂）
  - 添加 15% padding（避免裁切不完整）
  - 調整為正方形（避免變形）
- **輸出**: 裁切後的影片（聚焦簽名者，去除背景）

#### **2. 影片正規化（SlidingWindowInference）**
- **幀數調整策略**:
  - **< 80 幀 且 差距 ≤ 20 幀**:
    - 方法：線性插值補齊
    - 範例：75幀 → 80幀（插入5幀）
  - **< 80 幀 且 差距 > 20 幀**:
    - 方法：均勻重複填充
    - 範例：40幀 → 80幀（每幀重複2次）
  - **≥ 80 幀**:
    - 方法：保持原始長度或標準重採樣
    - 範例：200幀 → 200幀（保持）
- **解析度**: Resize 到 224×224
- **顏色空間**: BGR → RGB
- **FPS**: 標準化到 30 fps

### **窗口劃分**
- **窗口大小**: 80 幀（~2.67 秒 @ 30fps）
- **滑動步長**: 80 幀（無重疊，可調整為 40/60 幀）
- **窗口數量**: `(總幀數 - 80) ÷ 步長 + 1`
  - 範例 1：80 幀影片 → 1 個窗口
  - 範例 2：200 幀影片 → 2 個窗口
  - 範例 3：240 幀影片 → 3 個窗口

### **特徵提取**
- **特徵維度**: 300 幀 × 1119 維
  - **RGB 特徵**: 300×512（ResNet-50 提取）
  - **骨架特徵**: 300×607（MediaPipe Holistic）
    - 33 個身體關鍵點
    - 21 個左手關鍵點
    - 21 個右手關鍵點
    - 468 個臉部關鍵點
  - **合併**: 水平拼接 → 300×1119
- **處理速度**: ~2-3 秒/窗口（CPU）

## 模型資訊

- **架構**: BiGRU + 多頭注意力機制
- **輸入**: 300×1119 特徵矩陣
- **輸出**: 15 個手語單詞的信心度分佈
- **訓練集**: 15 個手語單詞，每個約 88 幀
- **準確率**: 
  - **訓練前（無智能裁切）**: 85-90%
  - **訓練後（智能裁切）**: 92-96% ⬆️ **+5-10%**
- **推論速度**: 
  - 特徵提取：~2-3 秒/窗口（CPU）
  - 模型推論：~200-500 毫秒/窗口
  - OpenAI 重組：~2-5 秒（取決於窗口數量）
- **信心度**: 70-85%（經過校準）

### **性能提升（智能裁切）**

| 指標 | Before（無裁切） | After（智能裁切） | 提升 |
|------|-----------------|------------------|------|
| 識別準確度 | 85-90% | **92-96%** | +5-10% |
| 背景干擾 | 大量 | **幾乎無** | ✅ |
| 手臂完整性 | 可能裁切 | **100% 保留** | ✅ |
| 處理時間 | ~20-35s | ~25-40s | +3-5s |

## 影片管理

### 自動保留策略
- 所有透過 Messenger Bot 收到的影片會**保留**在伺服器上
- 影片存放於 `DATA_DIR/downloaded_videos/` 目錄
- 重複影片會跳過下載（基於 MD5 哈希檢測）
- 已處理的影片可直接使用，無需重新識別

### 影片儲存空間估算
- 單個手語影片：5-50 MB（依長度而定）
- 建議預留空間：2-5 GB（約 100-200 個影片）

### 手動清理影片
如需清理舊影片，可透過以下方式：

#### 方法 1：刪除特定影片
```bash
# 在 Zeabur Dashboard 中進入 Container Shell
# 或使用 Zeabur CLI

# 刪除特定影片
cd /app/data/downloaded_videos
rm <video_hash>.mp4
```

#### 方法 2：清空所有影片
```bash
# 刪除所有影片
rm -rf /app/data/downloaded_videos/*

# 清空哈希記錄（重要！）
rm /app/data/downloaded_videos.json
rm /app/data/processed_count.json

# 重啟服務（在 Zeabur Dashboard 中）
```

#### 方法 3：設定自動清理（cron job）
```bash
# 每週日凌晨 2 點清理 30 天前的影片
0 2 * * 0 find /opt/render/project/data/downloaded_videos -name "*.mp4" -mtime +30 -delete
```

## 疑難排解

### 部署相關

#### 問題：Render 部署失敗，提示記憶體不足
**原因**：深度學習依賴包（TensorFlow、PyTorch）體積較大

**解決方案**：
1. 使用 Docker 部署（更穩定）
2. 升級到 Standard 或更高等級的 Instance Type
3. 如果使用 Python 部署，確保 Build 時間不超時（調整 Build Command）

#### 問題：模型載入失敗
**原因**：模型文件未正確上傳到 Git 倉庫

**解決方案**：
```bash
# 確認模型文件存在
ls -lh model_output/best_model_mps.keras
ls -lh model_output/label_map.json

# 如果文件過大（>100MB），使用 Git LFS
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add model_output/best_model_mps.keras
git commit -m "Add model with Git LFS"
git push

# 檢查 Zeabur 部署狀態
curl "https://signar.zeabur.app/debug"
```

#### 問題：OpenCV 或 MediaPipe 無法載入
**原因**：缺少系統依賴

**解決方案**：
- 使用提供的 Dockerfile（已包含所需依賴）
- 或在 Render Build Command 中添加：
```bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && pip install -r requirements.txt
```

### 識別相關

#### 問題：識別結果不準確
**可能原因**：
1. 影片品質不佳（模糊、光線不足）
2. 手語動作不完整或過快
3. 影片角度不正確

**建議**：
- 使用正面拍攝，確保手部和上半身清晰可見
- 每個手語動作持續 2-3 秒
- 充足光線，避免背光

#### 問題：識別速度過慢
**原因**：影片過長，窗口過多

**解決方案**：
1. 調整滑動步長（修改 `stride` 參數）
2. 升級 Render Instance Type（更多 CPU 核心）
3. 使用較短的影片（建議 10 秒以內）

#### 問題：OpenAI 句子重組失敗
**原因**：未設定 `OPENAI_API_KEY` 或 API 配額用盡

**解決方案**：
- 設定有效的 OpenAI API Key
- 或不使用 OpenAI（系統會自動使用 Top-1 單詞序列）

### Messenger Bot 相關

#### 問題：前端無法播放影片
**原因**：
- 影片路徑錯誤
- 檔案權限問題
- 瀏覽器不支援影片格式

**解決方案**：
1. 檢查瀏覽器控制台是否有 404 錯誤：`https://signar.zeabur.app/videos/<hash>`
2. 確認影片檔案存在：
```bash
# 在 Zeabur Container Shell 中檢查
ls -lh /app/data/downloaded_videos/
```
3. 檢查檔案權限：
```bash
chmod 644 /app/data/downloaded_videos/*.mp4
```

#### 問題：Messenger 無法回覆訊息
**原因**：
- `PAGE_ACCESS_TOKEN` 錯誤或過期
- 網路連線問題

**解決方案**：
1. 在 Facebook Developers Console 重新生成 Token
2. 確認 Render 伺服器能連接 `graph.facebook.com`：
```bash
curl https://graph.facebook.com/v18.0/me?access_token=YOUR_TOKEN
```
3. 查看 Render Logs 確認錯誤訊息

#### 問題：WebSocket 連線失敗
**原因**：防火牆或代理伺服器阻擋

**解決方案**：
- 確認 Zeabur 允許 WebSocket 連線（預設支援）
- 檢查前端控制台是否有 Socket.IO 錯誤：`https://signar.zeabur.app/`
- 嘗試重新整理頁面
- 檢查瀏覽器是否阻擋 WebSocket

### 效能優化

#### 提升識別速度
1. **調整窗口步長**：
```python
# 在 app.py 中修改
recognizer = SlidingWindowInference(
    stride=80  # 增加步長，減少窗口數量
)
```

2. **使用 GPU**（如果 Zeabur 支援）：
```python
recognizer = SlidingWindowInference(
    device='gpu'  # 使用 GPU 加速
)
```

3. **預載入模型**：
- 系統啟動時自動載入，無需每次重新載入

#### 降低記憶體使用
1. 定期清理舊影片
2. 使用較小的批次大小
3. 關閉不必要的日誌輸出

## 開發指南

### 本地開發

```bash
# 1. Clone 專案
git clone <your-repo-url>
cd 大專生院校

# 2. 建立虛擬環境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 設定環境變數
cp .env.example .env
# 編輯 .env 填入你的 Token

# 5. 啟動服務
python app.py
```

### 生產環境測試

```bash
# 測試生產環境
curl "https://signar.zeabur.app/health"
curl "https://signar.zeabur.app/debug"
curl "https://signar.zeabur.app/queue_status"
```

### 測試手語識別

```python
from sliding_window_inference import SlidingWindowInference

# 初始化識別器
recognizer = SlidingWindowInference(
    model_path='model_output/best_model_mps.keras',
    label_map_path='model_output/label_map.json',
    device='cpu',
    stride=80,
    openai_api_key='your-api-key'  # 選填
)

# 處理影片
results = recognizer.process_video('test_video.mp4')

# 獲取句子
if recognizer.openai_client:
    sentence, explanation = recognizer.compose_sentence_with_openai(results)
    print(f"識別結果: {sentence}")
```

### 生產環境測試範例

```python
# 使用生產環境 API 測試
import requests

# 檢查健康狀態
response = requests.get('https://signar.zeabur.app/health')
print(f"健康檢查: {response.json()}")

# 檢查隊列狀態
response = requests.get('https://signar.zeabur.app/queue_status')
print(f"隊列狀態: {response.json()}")
```

### 新增手語單詞

1. **準備訓練數據**：每個單詞至少 50 個影片樣本
2. **重新訓練模型**：使用 `training/train_bigrunet_tpu.py`
3. **更新標籤映射**：修改 `model_output/label_map.json`
4. **測試新模型**：確保準確率達標
5. **部署更新**：推送到 Git 倉庫，Render 自動重新部署

## 授權

MIT License

## 聯絡方式

如有問題或建議，請開 Issue 或 Pull Request。

---

## 生產環境

### 目前部署狀態
- **平台**: Zeabur
- **URL**: https://signar.zeabur.app/
- **狀態**: ✅ 正常運行
- **功能**: 完整的手語識別系統（智能裁切 + 排隊系統 + 多語言支援）

### 系統規格
- **技術堆疊**: Flask + Socket.IO + TensorFlow + MediaPipe
- **模型**: BiGRU + 注意力機制（15 個手語單詞）
- **準確度**: 92-96%（智能裁切後）
- **處理時間**: ~25-40 秒/影片
- **支援語言**: 繁中、英文、日文、韓文

### 聯絡測試
如需測試系統，請透過 Facebook Messenger 發送手語影片至已設定之粉絲專頁。

---

**部署完成後即可使用** 🚀🤟