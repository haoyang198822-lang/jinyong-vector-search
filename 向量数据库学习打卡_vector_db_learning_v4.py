#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库学习打卡 - 命令行版向量检索脚本
使用阿里云 text-embedding-v4 + Faiss 实现《倚天屠龙记》章节检索

启动方式: python vector_db_learning_v4.py

依赖: pip install openai faiss-cpu numpy
"""

import os
import json
import hashlib
import time
import re
import numpy as np
import faiss
from openai import OpenAI

# ============== 配置 ==============
DATA_SOURCE = "金庸-倚天屠龙记txt精校版 .txt"
OUTPUT_DIR = "."

# 向量配置
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024  # text-embedding-v4 输出维度
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 分块配置
CHUNK_SIZE = 400       # 每块字符数
CHUNK_OVERLAP = 100    # 重叠字符数
BATCH_SIZE = 10        # API批处理大小
API_INTERVAL = 0.1     # 每次API调用间隔（秒）

# 输出文件
INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss_index_v4.bin")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata_v4.json")
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings_v4.npy")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "query_results_v4.txt")
STATS_FILE = os.path.join(OUTPUT_DIR, "stats_v4.json")


def load_text():
    """加载文本文件（GBK编码）"""
    print(f"正在加载文本: {DATA_SOURCE}")
    with open(DATA_SOURCE, 'r', encoding='gbk', errors='ignore') as f:
        text = f.read()
    print(f"文本加载完成，总字符数: {len(text):,}")
    return text


def detect_chapters(text):
    """检测章节，返回章节列表[(起始位置, 章节号, 章节标题), ...]"""
    # 中文数字章节检测模式
    chapter_patterns = [
        r'\n第([一二三四五六七八九十百千零〇\d]+)章\s+(.+?)\n',
        r'\n第([一二三四五六七八九十百千零〇\d]+)节\s+(.+?)\n',
        r'\n第([一二三四五六七八九十百千零〇\d]+)回\s+(.+?)\n',
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
            
            # 转换中文数字为阿拉伯数字
            num = 0
            temp = 0
            for char in num_str:
                if char in '零〇':
                    continue
                elif char in '一二':
                    temp = chapter_num_map.get(char, 0)
                elif char == '十':
                    temp = temp * 10 + 10 if temp > 0 else 10
                elif char == '百':
                    temp = temp * 100 if temp > 0 else 100
                elif char in '三千':
                    temp = chapter_num_map.get(char, 0)
                elif char.isdigit():
                    temp = int(char)
                num += temp
                temp = 0
            
            if num > 0:
                chapters.append((start_pos, num, title))
    
    # 按位置排序
    chapters.sort(key=lambda x: x[0])
    
    # 过滤重复（保留最早的）
    seen = set()
    filtered = []
    for ch in chapters:
        if ch[1] not in seen:
            seen.add(ch[1])
            filtered.append(ch)
    
    print(f"检测到 {len(filtered)} 个章节")
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
        
        # 尝试找到句号断点
        if len(current_chunk) + line_len > chunk_size:
            # 查找最后一个句号
            last_period = current_chunk.rfind('。')
            if last_period > chunk_size * 0.6:  # 至少60%时断句
                # 截断到句号
                chunk_text_content = current_chunk[:last_period + 1]
                chunks.append({
                    'text': chunk_text_content,
                    'start_pos': chunk_start_pos,
                    'end_pos': chunk_start_pos + len(chunk_text_content),
                    'start_line': chunk_start_line,
                    'end_line': current_line - 1
                })
                # 重叠部分
                overlap_text = current_chunk[max(0, last_period - overlap):]
                chunk_start_pos = chunk_start_pos + last_period - overlap + 1
                chunk_start_line = current_line - 1
                current_chunk = overlap_text + '\n' + line
            else:
                # 强制断块
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
    
    # 添加最后一块
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'start_pos': chunk_start_pos,
            'end_pos': current_pos,
            'start_line': chunk_start_line,
            'end_line': current_line - 1
        })
    
    print(f"分块完成，共 {len(chunks)} 个文本块")
    return chunks


def compute_text_hash(text):
    """计算文本哈希"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]


def get_embedding_client():
    """创建OpenAI兼容客户端"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def get_embeddings_batch(client, texts, model=EMBEDDING_MODEL):
    """批量获取文本向量"""
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        for item in response.data:
            embeddings.append(item.embedding)
        if i + BATCH_SIZE < len(texts):
            time.sleep(API_INTERVAL)
    
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings):
    """构建Faiss索引"""
    # 归一化向量（用于内积相似度）
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    normalized = embeddings / norms
    
    # 创建内积索引
    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)
    
    print(f"Faiss索引构建完成，维度: {dim}, 数量: {index.ntotal}")
    return index, normalized


def search(index, query_embedding, k=5):
    """向量检索"""
    # 归一化查询向量
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm
    
    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
    similarities, indices = index.search(query_embedding, k)
    
    return similarities[0], indices[0]


def save_index(index, embeddings, metadata):
    """保存索引和元数据"""
    print("正在保存索引文件...")
    
    faiss.write_index(index, INDEX_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"索引已保存:")
    print(f"  - {INDEX_FILE}")
    print(f"  - {METADATA_FILE}")
    print(f"  - {EMBEDDINGS_FILE}")


def load_existing_index():
    """加载已有索引"""
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        print("检测到已有索引，正在加载...")
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"索引加载完成: {index.ntotal} 条记录")
        return index, metadata, embeddings
    return None, None, None


def main():
    """主流程"""
    print("=" * 50)
    print("向量数据库学习打卡 - 文本向量化检索")
    print("=" * 50)
    
    # 尝试加载已有索引
    index, metadata, embeddings = load_existing_index()
    
    if index is None:
        print("\n[步骤1] 加载文本...")
        text = load_text()
        
        print("\n[步骤2] 检测章节...")
        chapters = detect_chapters(text)
        
        print("\n[步骤3] 文本分块...")
        chunks = chunk_text(text)
        
        print("\n[步骤4] 生成元数据...")
        for i, chunk in enumerate(chunks):
            chapter_num, chapter_title = get_chapter_info(chunk['start_pos'], chapters)
            chunk['chunk_id'] = i
            chunk['chapter_num'] = chapter_num
            chunk['chapter_title'] = chapter_title
            chunk['char_count'] = len(chunk['text'])
            chunk['text_hash'] = compute_text_hash(chunk['text'])
        
        metadata = chunks
        
        print("\n[步骤5] 获取文本向量...")
        client = get_embedding_client()
        texts = [c['text'] for c in chunks]
        
        print(f"正在生成 {len(texts)} 个文本块的向量...")
        embeddings = get_embeddings_batch(client, texts)
        
        print(f"\n向量矩阵形状: {embeddings.shape}")
        
        print("\n[步骤6] 构建Faiss索引...")
        index, embeddings = build_faiss_index(embeddings)
        
        print("\n[步骤7] 保存索引...")
        save_index(index, embeddings, metadata)
        
        # 生成统计信息
        stats = {
            'total_chunks': len(chunks),
            'embedding_dim': EMBEDDING_DIM,
            'embedding_model': EMBEDDING_MODEL,
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'chapter_count': len(chapters),
            'file_list': [INDEX_FILE, METADATA_FILE, EMBEDDINGS_FILE]
        }
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n统计信息已保存: {STATS_FILE}")
    
    # ============== 测试查询 ==============
    print("\n" + "=" * 50)
    print("开始测试查询")
    print("=" * 50)
    
    client = get_embedding_client()
    
    test_queries = [
        "张无忌和赵敏的故事",
        "明教的高手有哪些",
        "武当派的张三丰",
        "倚天剑和屠龙刀的来历",
        "六大派围攻光明顶"
    ]
    
    results_all = []
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)
        
        # 获取查询向量
        query_embedding = get_embeddings_batch(client, [query])[0]
        
        # 检索
        similarities, indices = search(index, query_embedding, k=5)
        
        # 输出结果
        query_results = {
            'query': query,
            'results': []
        }
        
        for rank, (sim, idx) in enumerate(zip(similarities, indices)):
            if idx < len(metadata):
                chunk = metadata[idx]
                result = {
                    'rank': rank + 1,
                    'similarity': float(sim),
                    'chapter': f"第{chunk['chapter_num']}章 {chunk['chapter_title']}",
                    'line': f"{chunk['start_line']}-{chunk['end_line']}",
                    'preview': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                }
                query_results['results'].append(result)
                
                print(f"  [{rank+1}] 相似度: {sim:.4f} | {result['chapter']} | 行: {result['line']}")
                print(f"       {result['preview']}")
        
        results_all.append(query_results)
    
    # 保存查询结果
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    print(f"\n查询结果已保存: {RESULTS_FILE}")
    
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
