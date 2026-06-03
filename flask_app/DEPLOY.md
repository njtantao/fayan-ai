# 法眼AI 部署说明

## 方案一：Render（推荐，免费）

### 部署步骤

1. 访问 [render.com](https://render.com)，用 GitHub 账号登录

2. 点击 **New → Blueprint**，连接本仓库 `njtantao/fayan-ai`

3. Render 会读取 `render.yaml` 自动识别服务，点击 **Apply**

4. 在环境变量中填入：
   - `MINIMAX_API_KEY` = 你的 MiniMax API Key
   - `DATA_URL` = 案例库 CSV 的直链（如 GitHub Release 链接）

5. 点击 **Save Changes**，等待部署完成

6. 部署成功后访问 `https://fayan-ai.onrender.com`

---

### 方案二：手动部署到 Render

1. 登录 [render.com](https://render.com)
2. **New → Web Service**
3. 连接 GitHub 仓库 `njtantao/fayan-ai`
4. 设置：
   - **Root Directory**: `flask_app`（或留空，在根目录运行）
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. 添加环境变量：
   - `MINIMAX_API_KEY`
   - `DATA_URL`（可选）
6. 点击 **Create Web Service**

---

## 案例库说明

案例库 `data/all_cases_perfect.csv`（约50MB）太大，不适合直接上传 GitHub。

有两个方案：

**方案A**：上传到 GitHub Release，通过 `DATA_URL` 环境变量在启动时下载

**方案B**：使用云存储（如七牛云、阿里云OSS）直链

---

## 本地运行

```bash
cd flask_app
pip install -r requirements.txt
python app.py
# 访问 http://localhost:5099
```
