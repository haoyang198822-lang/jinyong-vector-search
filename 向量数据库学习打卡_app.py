#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库学习打卡 - Streamlit 交互式向量检索 Web 应用
使用阿里云 text-embedding-v4 + Faiss 实现《倚天屠龙记》章节检索

启动方式: 
  1. 命令行: cd 向量数据库学习打卡 && streamlit run app.py
  2. 或直接: streamlit run 向量数据库学习打卡/app.py

依赖: pip install streamlit openai faiss-cpu numpy
"""

import os
import sys
import json
import hashlib
import re
import time
import streamlit as st
import numpy as np
import faiss
from openai import OpenAI
import networkx as nx
from pyvis.network import Network

# ============== 配置 ==============
DATA_SOURCE = "金庸-倚天屠龙记txt精校版 .txt"

# 向量配置
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# LLM 配置 (RAG用)
LLM_MODEL = "qwen-plus"

# 分块配置
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
BATCH_SIZE = 10
API_INTERVAL = 0.1

# 输出文件
INDEX_FILE = "faiss_index_v4.bin"
METADATA_FILE = "metadata_v4.json"
EMBEDDINGS_FILE = "embeddings_v4.npy"


# ============== 核心函数 ==============

def load_text():
    """加载文本文件（GBK编码）"""
    with open(DATA_SOURCE, 'r', encoding='gbk', errors='ignore') as f:
        text = f.read()
    return text


def detect_chapters(text):
    """检测章节"""
    chapter_patterns = [
        r'\n第([一二三四五六七八九十百千零〇\d]+)章\s+(.+?)\n',
        r'\n第([一二三四五六七八九十百千零〇\d]+)节\s+(.+?)\n',
        r'\n第([一二三四五六七八九十百千零〇\d]+)回\s+(.+?)\n',
        r'\n([一二三四五六七八九十\d]+)\s+(.+?)\n',
    ]
    
    chapters = []
    chapter_num_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '百': 100, '千': 1000, '零': 0, '〇': 0
    }
    
    for pattern in chapter_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            start_pos = match.start()
            num_str = match.group(1)
            title = match.group(2).strip()
            
            if num_str.isdigit():
                num = int(num_str)
            else:
                num = 0
                temp = 0
                for char in num_str:
                    if char in '零〇':
                        continue
                    elif char in chapter_num_map and char not in '十百千':
                        temp = chapter_num_map.get(char, 0)
                    elif char == '十':
                        temp = temp * 10 + 10 if temp > 0 else 10
                    elif char == '百':
                        temp = temp * 100 if temp > 0 else 100
                    elif char == '千':
                        temp = chapter_num_map.get(char, 0)
                    elif char.isdigit():
                        temp = int(char)
                    num += temp
                    temp = 0
            
            if num > 0:
                chapters.append((start_pos, num, title))
    
    chapters.sort(key=lambda x: x[0])
    seen = set()
    filtered = []
    for ch in chapters:
        if ch[1] not in seen:
            seen.add(ch[1])
            filtered.append(ch)
    
    return filtered


def get_chapter_info(pos, chapters):
    """根据位置获取所在章节信息"""
    chapter_num = 1
    chapter_title = "序章"
    for i, (start, num, title) in enumerate(chapters):
        if pos >= start:
            chapter_num = num
            chapter_title = title
        else:
            break
    return chapter_num, chapter_title


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """文本分块，句号边界优化"""
    chunks = []
    lines = text.split('\n')
    
    current_pos = 0
    current_line = 0
    current_chunk = ""
    chunk_start_pos = 0
    chunk_start_line = 0
    
    while current_line < len(lines):
        line = lines[current_line]
        line_len = len(line)
        
        if len(current_chunk) + line_len > chunk_size:
            last_period = current_chunk.rfind('。')
            if last_period > chunk_size * 0.6:
                chunk_text_content = current_chunk[:last_period + 1]
                chunks.append({
                    'text': chunk_text_content,
                    'start_pos': chunk_start_pos,
                    'end_pos': chunk_start_pos + len(chunk_text_content),
                    'start_line': chunk_start_line,
                    'end_line': current_line - 1
                })
                overlap_text = current_chunk[max(0, last_period - overlap):]
                chunk_start_pos = chunk_start_pos + last_period - overlap + 1
                chunk_start_line = current_line - 1
                current_chunk = overlap_text + '\n' + line
            else:
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_pos': chunk_start_pos,
                    'end_pos': chunk_start_pos + len(current_chunk),
                    'start_line': chunk_start_line,
                    'end_line': current_line - 1
                })
                chunk_start_pos = current_pos
                chunk_start_line = current_line
                current_chunk = line
        else:
            current_chunk += '\n' + line
        
        current_pos += line_len + 1
        current_line += 1
    
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'start_pos': chunk_start_pos,
            'end_pos': current_pos,
            'start_line': chunk_start_line,
            'end_line': current_line - 1
        })
    
    return chunks


def compute_text_hash(text):
    """计算文本哈希"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]


def get_embedding_client(api_key):
    """创建OpenAI兼容客户端"""
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def get_embeddings_batch(client, texts, model=EMBEDDING_MODEL, progress_callback=None):
    """批量获取文本向量"""
    embeddings = []
    total = len(texts)
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        for item in response.data:
            embeddings.append(item.embedding)
        
        if progress_callback:
            progress_callback(min(i + BATCH_SIZE, total), total)
        
        if i + BATCH_SIZE < len(texts):
            time.sleep(API_INTERVAL)
    
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings):
    """构建Faiss索引"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    
    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)
    
    return index, normalized


def search(index, query_embedding, k=5):
    """向量检索"""
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm

    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    similarities, indices = index.search(query_embedding, k)

    return similarities[0], indices[0]


def rag_answer(client, question, retrieved_chunks, model=LLM_MODEL):
    """基于检索结果生成回答 (RAG)"""
    context = "\n\n".join([
        f"【片段{i+1}】（第{chunk['chapter_num']}章 {chunk['chapter_title']} 第{chunk['start_line']}-{chunk['end_line']}行）:\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])

    system_prompt = """你是一个精通《倚天屠龙记》的文学助手。你的任务是基于给定的原文片段回答用户的问题。

要求：
1. 只基于提供的原文片段回答，不要编造内容
2. 如果片段不足以回答，请明确说明"根据提供的信息无法回答"
3. 回答要条理清晰，适当引用原文
4. 用中文回答，语言流畅自然
5. 如果涉及多个片段，可以综合整理后回答"""

    user_prompt = f"""请基于以下原文片段回答问题：

{context}

---

问题：{question}

回答："""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成回答失败: {str(e)}"


# ============== 知识图谱数据 ==============
CHARACTERS = [
    # 主要人物
    "张无忌", "赵敏", "周芷若", "殷素素", "张翠山", "张三丰", "谢逊",
    # 明教
    "杨逍", "范遥", "韦一笑", "殷天正", "黛绮丝", "小昭",
    # 武当
    "宋远桥", "俞莲舟", "俞岱岩", "张松溪", "殷梨亭", "莫声谷",
    # 峨眉
    "灭绝师太", "丁敏君",
    # 少林
    "空闻", "空智", "空性",
    # 其他
    "成昆", "陈友谅", "汝阳王", "王保保"
]

FACTIONS = [
    "明教", "武当派", "峨眉派", "少林派", "丐帮", "天鹰教", "汝阳王府"
]

MARTIAL_ARTS = [
    "九阳神功", "乾坤大挪移", "太极拳", "太极剑", "玄冥神掌",
    "九阴白骨爪", "降龙十八掌", "少林龙爪手", "七伤拳",
    "峨嵋九阳功", "兰花拂穴手", "无声无色", "壁虎游墙功"
]

RELATIONSHIPS = [
    # 人物-人物关系
    ("张无忌", "爱慕", "赵敏"),
    ("张无忌", "爱慕", "周芷若"),
    ("张无忌", "义父", "谢逊"),
    ("张无忌", "父亲", "张翠山"),
    ("张无忌", "母亲", "殷素素"),
    ("张无忌", "师祖", "张三丰"),
    ("张翠山", "妻子", "殷素素"),
    ("张翠山", "师父", "张三丰"),
    ("殷素素", "丈夫", "张翠山"),
    ("赵敏", "爱慕", "张无忌"),
    ("周芷若", "爱慕", "张无忌"),
    ("谢逊", "义子", "张无忌"),
    ("张三丰", "徒弟", "宋远桥"),
    ("张三丰", "徒弟", "俞莲舟"),
    ("张三丰", "徒弟", "张翠山"),
    ("张三丰", "徒孙", "张无忌"),
    ("杨逍", "下属", "张无忌"),
    ("范遥", "下属", "张无忌"),
    ("韦一笑", "下属", "张无忌"),
    ("殷天正", "下属", "张无忌"),
    ("灭绝师太", "徒弟", "周芷若"),
    ("灭绝师太", "徒弟", "丁敏君"),
    ("成昆", "仇人", "谢逊"),
    ("汝阳王", "女儿", "赵敏"),
    ("汝阳王", "儿子", "王保保"),
    
    # 人物-门派关系
    ("张无忌", "属于", "明教"),
    ("张无忌", "属于", "武当派"),
    ("赵敏", "属于", "汝阳王府"),
    ("周芷若", "属于", "峨眉派"),
    ("张三丰", "属于", "武当派"),
    ("谢逊", "属于", "明教"),
    ("杨逍", "属于", "明教"),
    ("范遥", "属于", "明教"),
    ("韦一笑", "属于", "明教"),
    ("殷天正", "属于", "明教"),
    ("殷素素", "属于", "天鹰教"),
    ("张翠山", "属于", "武当派"),
    ("宋远桥", "属于", "武当派"),
    ("俞莲舟", "属于", "武当派"),
    ("灭绝师太", "属于", "峨眉派"),
    ("丁敏君", "属于", "峨眉派"),
    ("空闻", "属于", "少林派"),
    ("空智", "属于", "少林派"),
    ("空性", "属于", "少林派"),
    ("成昆", "属于", "少林派"),
    ("陈友谅", "属于", "丐帮"),
    
    # 人物-武功关系
    ("张无忌", "修炼", "九阳神功"),
    ("张无忌", "修炼", "乾坤大挪移"),
    ("张无忌", "修炼", "太极拳"),
    ("张无忌", "修炼", "太极剑"),
    ("张三丰", "创编", "太极拳"),
    ("张三丰", "创编", "太极剑"),
    ("张三丰", "修炼", "九阳神功"),
    ("张翠山", "修炼", "太极拳"),
    ("谢逊", "修炼", "七伤拳"),
    ("成昆", "修炼", "幻阴指"),
    ("成昆", "修炼", "少林九阳功"),
    ("杨逍", "修炼", "乾坤大挪移"),
    ("殷天正", "修炼", "鹰爪功"),
    ("灭绝师太", "修炼", "峨嵋九阳功"),
    ("周芷若", "修炼", "九阴白骨爪"),
    ("赵敏", "修炼", "玄冥神掌"),
    ("韦一笑", "修炼", "寒冰绵掌"),
    ("宋远桥", "修炼", "太极拳"),
    ("俞莲舟", "修炼", "太极拳"),
    
    # 门派-门派关系
    ("明教", "敌对", "武当派"),
    ("明教", "敌对", "峨眉派"),
    ("明教", "敌对", "少林派"),
    ("明教", "敌对", "丐帮"),
    ("武当派", "友好", "少林派"),
    ("峨眉派", "友好", "少林派"),
    ("汝阳王府", "敌对", "明教"),
]

def build_knowledge_graph():
    """构建知识图谱"""
    G = nx.Graph()
    
    # 添加人物节点
    for char in CHARACTERS:
        G.add_node(char, type='character', color='#E8E8E8', size=20)
    
    # 添加门派节点
    for faction in FACTIONS:
        G.add_node(faction, type='faction', color='#D4E5E5', size=30)
    
    # 添加武功节点
    for art in MARTIAL_ARTS:
        G.add_node(art, type='martial_art', color='#F5E6D3', size=15)
    
    # 添加关系边
    for source, relation, target in RELATIONSHIPS:
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target, label=relation)
    
    return G

def generate_graph_html(graph):
    """生成图谱HTML"""
    net = Network(height="600px", width="100%", directed=True, bgcolor="#1a1a1a")
    
    for node in graph.nodes(data=True):
        node_name, attrs = node
        node_type = attrs.get('type', 'character')
        
        if node_type == 'character':
            color = '#ffffff'
            font_color = '#1a1a1a'
            size = 20
        elif node_type == 'faction':
            color = '#a0d2db'
            font_color = '#1a1a1a'
            size = 30
        else:  # martial_art
            color = '#f0d9b5'
            font_color = '#1a1a1a'
            size = 15
        
        net.add_node(
            node_name,
            label=node_name,
            color=color,
            size=size,
            font={'size': 14, 'color': font_color},
            shape='circle',
            borderWidth=0,
            shadow=False
        )
    
    for edge in graph.edges(data=True):
        source, target, attrs = edge
        label = attrs.get('label', '')
        net.add_edge(source, target, label=label, font={'size': 10, 'color': '#ffffff', 'strokeWidth': 0}, color='#666666')
    
    # 设置布局
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 100,
          "updateInterval": 25
        },
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 250,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": false,
        "hideEdgesOnZoom": false
      },
      "nodes": {
        "borderWidth": 0,
        "shadow": false,
        "color": {
          "background": "#ffffff",
          "border": "#1a1a1a",
          "highlight": {
            "background": "#4a90d9",
            "border": "#4a90d9"
          }
        }
      },
      "edges": {
        "smooth": {
          "enabled": true
        },
        "color": {
          "color": "#666666",
          "highlight": "#888888",
          "hover": "#888888"
        }
      }
    }
    """)
    
    # 保存到临时文件
    html_path = "/tmp/kg_graph.html"
    net.write_html(html_path)
    return html_path


def save_index(index, embeddings, metadata):
    """保存索引和元数据"""
    faiss.write_index(index, INDEX_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_existing_index():
    """加载已有索引"""
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        embeddings = np.load(EMBEDDINGS_FILE)
        return index, metadata, embeddings
    return None, None, None


def build_vector_index(progress_callback=None):
    """构建向量索引"""
    text = load_text()
    chapters = detect_chapters(text)
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        chapter_num, chapter_title = get_chapter_info(chunk['start_pos'], chapters)
        chunk['chunk_id'] = i
        chunk['chapter_num'] = chapter_num
        chunk['chapter_title'] = chapter_title
        chunk['char_count'] = len(chunk['text'])
        chunk['text_hash'] = compute_text_hash(chunk['text'])
    
    metadata = chunks
    texts = [c['text'] for c in chunks]
    
    # 获取向量
    client = get_embedding_client(st.session_state.api_key)
    
    def progressWrapper(current, total):
        if progress_callback:
            progress_callback(int(current * 0.8), total)  # 向量化占80%
    
    embeddings = get_embeddings_batch(client, texts, progress_callback=progressWrapper)
    index, embeddings = build_faiss_index(embeddings)
    
    def finalProgress(current, total):
        if progress_callback:
            progress_callback(int(total * 0.8 + current * 0.2), total)
    
    if progress_callback:
        finalProgress(100, 100)
    
    save_index(index, embeddings, metadata)
    
    return index, metadata, embeddings


# ============== Streamlit 应用 ==============

def init_session_state():
    """初始化会话状态"""
    defaults = {
        'index': None,
        'metadata': None,
        'embeddings': None,
        'query_history': [],
        'search_results': None,
        'last_query': None,
        'rag_question': None,
        'retrieved_chunks': None,
        'rag_answer': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar_config():
    """侧边栏配置"""
    st.sidebar.title("配置")
    
    env_api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    if env_api_key:
        st.session_state.api_key = env_api_key
        st.sidebar.success("API Key 已从环境变量加载")
    else:
        st.session_state.api_key = st.sidebar.text_input(
            "DASHSCOPE_API_KEY",
            value="",
            type="password",
            help="阿里云 DashScope API Key\n建议通过环境变量设置"
        )
        st.sidebar.info("export DASHSCOPE_API_KEY=your_key")
    
    if not st.session_state.api_key:
        st.sidebar.warning("请设置 API Key")
        return False
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("向量库主要信息")
    
    if st.session_state.index is None:
        index, metadata, _ = load_existing_index()
        if index is not None:
            st.session_state.index = index
            st.session_state.metadata = metadata
            st.session_state.embeddings = _
    
    if st.session_state.metadata:
        chapters = set(m['chapter_num'] for m in st.session_state.metadata)
        chapter_list = sorted(chapters)
        st.sidebar.metric("Chunk 数", len(st.session_state.metadata))
        st.sidebar.metric("章节数", len(chapters))
        st.sidebar.text(f"章节索引: {chapter_list[0]}-{chapter_list[-1]}" if chapter_list else "无")
    elif st.session_state.index is not None:
        st.sidebar.metric("记录数", st.session_state.index.ntotal)
        st.sidebar.metric("维度", st.session_state.index.d)
    else:
        st.sidebar.info("暂无索引")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("一键构建向量库")
    
    if st.session_state.index is None:
        if st.sidebar.button("构建向量库", type="primary", use_container_width=True):
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            try:
                def progress_callback(current, total):
                    progress = int(current / total * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"进度: {progress}%")
                
                with st.spinner("正在构建向量库..."):
                    index, metadata, embeddings = build_vector_index(progress_callback)
                    st.session_state.index = index
                    st.session_state.metadata = metadata
                    st.session_state.embeddings = embeddings
                
                progress_bar.empty()
                status_text.empty()
                st.sidebar.success(f"构建完成！共 {len(metadata)} 条")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"构建失败: {str(e)}")
    else:
        st.sidebar.success("向量库已就绪")
    
    st.sidebar.markdown("---")
    st.session_state.top_k = st.sidebar.slider(
        "Top-K 数量",
        min_value=1,
        max_value=20,
        value=5,
        help="返回最相似的K条结果"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("示例查询")
    
    preset_queries = [
        "张无忌和赵敏的故事",
        "明教的高手有哪些",
        "武当派的张三丰",
        "倚天剑和屠龙刀的来历",
        "六大派围攻光明顶"
    ]
    
    for query in preset_queries:
        if st.sidebar.button(query, key=f"preset_{query}", use_container_width=True):
            st.session_state.last_query = query
            st.rerun()
    
    return True


def main_content():
    """主内容区域"""
    st.title("倚天屠龙记 - 向量检索系统")
    st.markdown("基于 **text-embedding-v4** + **Faiss** 的智能文本检索")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["语义检索", "RAG智能问答", "知识图谱"])
    
    with tab1:
        search_tab()
    
    with tab2:
        rag_tab()
    
    with tab3:
        kg_tab()
    
    st.markdown("---")
    st.subheader("查询历史")
    
    if st.session_state.query_history:
        cols = st.columns(2)
        for i, hist in enumerate(reversed(st.session_state.query_history[-6:])):
            with cols[i % 2]:
                if st.button(f"{hist[:25]}{'...' if len(hist) > 25 else ''}", 
                           key=f"history_{i}", use_container_width=True):
                    st.session_state.last_query = hist
                    st.rerun()
    else:
        st.info("暂无查询历史")


def search_tab():
    """语义检索标签页"""
    st.markdown("### 语义检索模式")
    st.caption("基于向量相似度，直接返回最相关的文本片段")
    
    query = st.text_input(
        "输入查询内容",
        value=st.session_state.get('last_query', ''),
        placeholder="例如：张无忌修炼九阳神功",
        help="输入问题进行语义检索"
    )
    
    if query and query != st.session_state.get('last_query'):
        st.session_state.last_query = query
    
    search_clicked = st.button("检索", type="primary") if query else False
    
    if (search_clicked or st.session_state.search_results) and query:
        if st.session_state.index is None:
            st.error("请先构建向量库")
        else:
            with st.spinner("正在检索..."):
                try:
                    client = get_embedding_client(st.session_state.api_key)
                    query_embedding = get_embeddings_batch(client, [query])[0]
                    similarities, indices = search(st.session_state.index, query_embedding, 
                                                  k=st.session_state.top_k)
                    
                    results = []
                    for sim, idx in zip(similarities, indices):
                        if idx < len(st.session_state.metadata):
                            chunk = st.session_state.metadata[idx]
                            results.append({
                                'similarity': float(sim),
                                'chapter_num': chunk['chapter_num'],
                                'chapter_title': chunk['chapter_title'],
                                'start_line': chunk['start_line'],
                                'end_line': chunk['end_line'],
                                'text': chunk['text'],
                                'char_count': chunk['char_count']
                            })
                    
                    st.session_state.search_results = results
                    
                    if query not in st.session_state.query_history:
                        st.session_state.query_history.append(query)
                    
                    st.session_state.last_query = None
                    
                except Exception as e:
                    st.error(f"❌ 检索失败: {str(e)}")
    
    if st.session_state.search_results:
        st.markdown("---")
        st.subheader(f"📋 检索结果 (Top {len(st.session_state.search_results)})")
        
        for i, result in enumerate(st.session_state.search_results):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    similarity_percent = result['similarity'] * 100
                    st.markdown(f"**#{i+1}** | 相似度: `{similarity_percent:.2f}%`")
                    st.markdown(f"📖 {result['chapter_title']} | 行: {result['start_line']}-{result['end_line']}")
                
                with col2:
                    st.metric("相似度", f"{similarity_percent:.1f}%")
                
                with st.expander("📄 查看原文", expanded=i < 2):
                    st.text_area(
                        "原文内容",
                        result['text'],
                        height=150,
                        key=f"text_{i}",
                        disabled=True
                    )
                
                st.markdown("---")


def rag_tab():
    """RAG智能问答标签页"""
    st.markdown("### RAG智能问答模式")
    st.caption("基于检索增强生成，结合LLM理解后给出自然语言回答")
    
    if st.session_state.index is None:
        st.warning("请先在侧边栏构建向量库")
        return
    
    question = st.text_input(
        "输入您的问题",
        value=st.session_state.get('rag_question', ''),
        placeholder="例如：张无忌为什么会修炼九阳神功？",
        help="输入关于倚天屠龙记的问题"
    )
    
    if question and question != st.session_state.get('rag_question'):
        st.session_state.rag_question = question
    
    col1, col2 = st.columns([1, 3])
    with col1:
        retrieve_k = st.number_input("检索片段数", min_value=3, max_value=10, value=5)
    
    ask_clicked = st.button("提问", type="primary") if question else False
    
    if ask_clicked and question:
        with st.spinner("正在处理..."):
            try:
                client = get_embedding_client(st.session_state.api_key)
                query_embedding = get_embeddings_batch(client, [question])[0]
                similarities, indices = search(st.session_state.index, query_embedding, k=retrieve_k)
                
                retrieved_chunks = []
                for sim, idx in zip(similarities, indices):
                    if idx < len(st.session_state.metadata):
                        chunk = st.session_state.metadata[idx].copy()
                        chunk['similarity'] = float(sim)
                        retrieved_chunks.append(chunk)
                
                st.session_state.retrieved_chunks = retrieved_chunks
                
                answer = rag_answer(client, question, retrieved_chunks)
                st.session_state.rag_answer = answer
                
                if question not in st.session_state.query_history:
                    st.session_state.query_history.append(question)
                st.session_state.rag_question = None
                
            except Exception as e:
                st.error(f"处理失败: {str(e)}")
                return
    
    if st.session_state.rag_answer:
        st.markdown("---")
        st.subheader("回答")
        st.markdown(st.session_state.rag_answer)
        
        st.markdown("---")
        st.subheader("参考来源")
        
        retrieved_chunks = st.session_state.retrieved_chunks
        for i, chunk in enumerate(retrieved_chunks):
            sim_percent = chunk['similarity'] * 100
            with st.expander(f"片段{i+1}：{chunk['chapter_title']} (相似度: {sim_percent:.1f}%)"):
                st.text_area(
                    "原文",
                    chunk['text'],
                    height=120,
                    key=f"rag_chunk_{i}",
                    disabled=True
                )
        
        st.caption("回答基于检索到的原文片段生成，仅供参考")
        
        if st.button("清除回答"):
            st.session_state.rag_answer = None
            st.session_state.retrieved_chunks = None
            st.rerun()


def kg_tab():
    """知识图谱标签页"""
    st.markdown("### 知识图谱")
    st.caption("展示《倚天屠龙记》中的人物、门派和武功关系")
    
    # 筛选选项
    filter_options = ["全部", "仅人物", "仅门派", "仅武功", "人物与门派", "人物与武功"]
    selected_filter = st.selectbox("筛选显示", filter_options)
    
    with st.spinner("正在生成图谱..."):
        # 构建图谱
        G = build_knowledge_graph()
        
        # 应用筛选
        if selected_filter == "仅人物":
            nodes_to_keep = [n for n in G.nodes if G.nodes[n].get('type') == 'character']
            G = G.subgraph(nodes_to_keep)
        elif selected_filter == "仅门派":
            nodes_to_keep = [n for n in G.nodes if G.nodes[n].get('type') == 'faction']
            G = G.subgraph(nodes_to_keep)
        elif selected_filter == "仅武功":
            nodes_to_keep = [n for n in G.nodes if G.nodes[n].get('type') == 'martial_art']
            G = G.subgraph(nodes_to_keep)
        elif selected_filter == "人物与门派":
            nodes_to_keep = [n for n in G.nodes if G.nodes[n].get('type') in ['character', 'faction']]
            G = G.subgraph(nodes_to_keep)
        elif selected_filter == "人物与武功":
            nodes_to_keep = [n for n in G.nodes if G.nodes[n].get('type') in ['character', 'martial_art']]
            G = G.subgraph(nodes_to_keep)
        
        # 生成HTML
        html_path = generate_graph_html(G)
        
        # 显示图谱
        st.markdown("---")
        st.components.v1.html(open(html_path, 'r', encoding='utf-8').read(), height=650)
    
    # 图例
    st.markdown("---")
    st.markdown("#### 图例")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**人物**")
        st.markdown('<div style="background-color: #ffffff; width: 20px; height: 20px; border-radius: 50%; display: inline-block;"></div> 白色圆形节点', unsafe_allow_html=True)
    with col2:
        st.markdown("**门派**")
        st.markdown('<div style="background-color: #a0d2db; width: 25px; height: 25px; border-radius: 50%; display: inline-block;"></div> 浅蓝色圆形节点', unsafe_allow_html=True)
    with col3:
        st.markdown("**武功**")
        st.markdown('<div style="background-color: #f0d9b5; width: 18px; height: 18px; border-radius: 50%; display: inline-block;"></div> 浅棕圆形节点', unsafe_allow_html=True)
    
    st.caption("提示：拖拽节点可调整位置，滚轮可缩放，悬停查看关系")


def main():
    """主函数"""
    st.set_page_config(
        page_title="倚天屠龙记 - 向量检索",
        layout="wide"
    )
    
    init_session_state()
    
    if sidebar_config():
        main_content()
    else:
        st.warning("请在侧边栏配置 API Key")


if __name__ == "__main__":
    main()
