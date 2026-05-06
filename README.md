# 倚天屠龙记 - 向量检索与知识图谱系统

基于《倚天屠龙记》小说构建的向量检索、RAG问答和知识图谱可视化系统。

## 功能特性

- 🔍 **语义检索**：基于向量相似度搜索小说原文片段
- 🤖 **RAG 智能问答**：结合检索片段和 LLM 生成自然语言回答
- 📊 **知识图谱**：可视化展示人物、门派、武功的关系网
- 📈 **一键构建向量库**：支持进度展示，无需手动预处理

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 向量模型 | 阿里云 text-embedding-v4 |
| 向量索引 | Faiss |
| Web 框架 | Streamlit |
| 可视化 | NetworkX + PyVis |
| LLM | Qwen Plus (RAG用) |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

设置环境变量：

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

或在应用侧边栏输入 API Key。

### 3. 启动应用

```bash
streamlit run app.py
```

访问 http://localhost:8501 使用。

## 使用说明

### 语义检索模式

1. 在侧边栏点击「构建向量库」（首次使用）
2. 输入查询问题或点击示例查询
3. 调整「Top-K」参数修改返回结果数
4. 查看检索到的原文片段

### RAG 智能问答模式

1. 切换到「RAG智能问答」标签页
2. 输入问题后按回车直接生成回答
3. 查看参考来源片段

### 知识图谱模式

1. 切换到「知识图谱」标签页
2. 使用筛选器显示特定类型的关系
3. 拖拽节点调整位置，滚轮缩放
4. 悬停查看关系标签

## 项目结构

```
jinyong-vector-search/
├── app.py                  # Streamlit 主应用
├── vector_db_learning.py    # 命令行向量检索工具
├── requirements.txt         # Python 依赖
├── report.md               # 学习报告
├── yitian_tulongji.txt     # 小说文本
└── .gitignore             # Git 忽略文件
```

## 说明

- 向量库构建需要 API 调用，首次运行约需 1-2 分钟
- 生成的索引文件会被 `.gitignore` 排除，需在本地重建
- 知识图谱的节点数据为手动定义的典型关系

## 许可证

MIT License
