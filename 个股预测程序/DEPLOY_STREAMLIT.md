# 在线部署到 GitHub + Streamlit Community Cloud（手机可直接使用）

本项目的所有代码**本地版与线上版共用同一套** `core/` 计算内核，
因此不存在"两份逻辑不一致"的问题。线上只需要一个仓库 + 一个 `app.py`。

```
个股预测程序/                 ← 整个文件夹就是一个可部署仓库
├── app.py                   ★ Streamlit 入口（main file path 填这个）
├── run_local.py             本地命令行入口（不需要 streamlit）
├── core/                    计算内核（本地/线上共用）
├── requirements.txt         ★ 云端依赖
├── packages.txt             ★ 云端系统依赖（中文字体）
├── .streamlit/config.toml   主题与服务器配置
├── verify_all.py            自检脚本（CI/本地均可跑）
└── README.md
```

---

## 一、推送到 GitHub

### 1. 在 GitHub 上新建仓库

打开 <https://github.com/new>：

* **Repository name**：`stock-predictor`（或任意名字）
* **Public** ✅（Streamlit Community Cloud 免费版要求公开仓库；
  若要私有用，需付费或改用其他托管）
* 不要勾选 "Add a README file"（本地已有）

### 2. 在本地初始化并推送

在本文件夹（`个股预测程序`）下打开 PowerShell：

```powershell
git init
git add .
git commit -m "feat: 个股涨跌概率预测程序（8周期/8维度/多理论模型）"
git branch -M main
git remote add origin https://github.com/<你的用户名>/stock-predictor.git
git push -u origin main
```

> 首次推送会要求登录。推荐用 **Personal Access Token** 作为密码
> （GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens，
> 勾选该仓库的 `Contents: Read and write`）。

如果不希望命令行操作，也可以直接在 GitHub 网页上 **Add file → Upload files**
把整个文件夹拖进去（注意 `.streamlit/` 与 `core/` 目录要一起上传）。

---

## 二、部署到 Streamlit Community Cloud

1. 打开 <https://share.streamlit.io/>，用 GitHub 账号登录。
2. 点 **Create app** → 选 **Deploy a public app from GitHub**。
3. 填写：

   | 字段 | 值 |
   |---|---|
   | Repository | `<你的用户名>/stock-predictor` |
   | Branch | `main` |
   | **Main file path** | `app.py` |
   | App URL | 自定义子域名，如 `stock-predictor-xxx` |

4. 点 **Deploy**。首次构建约 2~4 分钟（会自动安装 `requirements.txt` 与 `packages.txt`）。

### 构建失败的常见原因

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: core` | 上传时漏了 `core/` 目录 | 确认仓库根目录有 `core/__init__.py` |
| 中文显示成方框 | 缺中文字体 | 确认 `packages.txt` 内容为 `fonts-noto-cjk` |
| `akshare` 安装超时 | akshare 依赖较多 | 可把 `requirements.txt` 里的 akshare 注释掉，程序会自动降级到腾讯/新浪/东财直连接口 |
| 数据全部取不到 | 云端出口 IP 被数据源限制 | 勾选页面侧边栏的**演示模式**先验证界面；数据源健康检查可定位具体是哪个源 |

---

## 三、手机使用

1. 部署完成后会得到一个固定网址，形如
   `https://stock-predictor-xxx.streamlit.app`。
2. 手机浏览器直接打开即可（页面已做移动端适配：卡片式指标、可横向滚动的表格、
   折叠式明细）。
3. **添加到主屏幕**（推荐，用起来接近原生 App）：
   * iOS Safari：分享按钮 → **添加到主屏幕**
   * Android Chrome：⋮ → **添加到主屏幕 / 安装应用**
4. 手机端使用建议：
   * 侧边栏默认收起，点左上角 `>` 展开；
   * 输入代码后点 **开始分析**；首次分析约 10~60 秒（取决于数据源响应）；
   * 若只想快速看结论，可关闭 **启用 AI 模型**（速度快很多）；
   * 结果页 **📥 导出/工具** 标签里可下载 HTML/文本报告到手机。

---

## 四、其他托管方式（可选）

| 方式 | 说明 |
|---|---|
| **Hugging Face Spaces** | 新建 Space → SDK 选 **Streamlit** → 上传同样文件（`app.py` 放根目录）。免费、国内访问通常比 streamlit.app 稳定 |
| **自建服务器 / NAS** | `pip install -r requirements.txt` 后执行<br>`streamlit run app.py --server.address 0.0.0.0 --server.port 8501`<br>手机与电脑同一局域网即可访问 `http://<电脑IP>:8501` |
| **内网穿透** | 配合 `frp` / `cloudflared` / `ngrok` 可把本地服务暴露到公网，**注意加访问口令** |

---

## 五、更新线上版本

```powershell
git add .
git commit -m "fix: ..."
git push
```

Streamlit Cloud 会自动检测到 push 并重建（约 1~2 分钟）。
若没有自动重建，在 <https://share.streamlit.io/> 的 app 菜单里点 **Reboot**。

---

## 六、安全与合规提醒

* 本程序**不收集也不上传**任何用户数据：所有计算都在服务端内存中完成，
  不写数据库、不做用户追踪。
* 若部署到公网，**不要**在代码或 `secrets.toml` 里放任何交易账号、
  券商 API Key 等敏感信息。
* 数据来自公开行情接口，可能存在延迟、缺失或错误；
  程序输出的一切概率均为统计推断，**不构成投资建议**。
