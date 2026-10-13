# Fog Detection RAG Assistant

面向雾天目标检测课程报告的检索增强生成（RAG）知识库助手。项目将支持 PDF 文本解析、中文向量检索、基于报告内容的问答、来源页码引用和检索效果评测。

## 当前状态

- [x] 建立项目骨架
- [ ] 解析 PDF 并保留页码元数据
- [ ] 建立向量知识库
- [ ] 接入大模型生成与来源引用
- [ ] 完成 Streamlit 交互界面
- [ ] 建立评测集并报告检索指标

## 目录结构

```text
.
├── assets/          # README 截图和演示素材
├── data/            # 本地原始文档（PDF 不提交到 GitHub）
├── evaluation/      # 评测问题、标注和评测脚本
├── src/             # PDF 解析、检索和生成模块
├── app.py           # Streamlit 应用入口
└── requirements.txt
```

## 本地运行

1. 使用 Python 3.11 创建并激活虚拟环境。
2. 安装依赖：`python -m pip install -r requirements.txt`。
3. 将 `.env.example` 复制为 `.env`，填写本地 API 配置。
4. 按照 `data/README.md` 放置待解析的 PDF。
5. 启动应用：`streamlit run app.py`。

## 隐私说明

原始课程报告可能包含个人或小组信息，因此不会直接提交到公开仓库。公开数据前应先完成脱敏检查。

