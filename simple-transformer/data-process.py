import pandas as pd
import numpy as np
import torch
import math

# 加载behaviors数据
def load_behaviors(file_path):
    behaviors = pd.read_csv(file_path, sep='\t', header=None, names=["impression_id", "user_id", "time", "history", "impressions"])
    return behaviors

# 加载news数据
def load_news(file_path):
    news = pd.read_csv(file_path, sep='\t', header=None, names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])
    return news

# 加载嵌入数据
def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            entity_id = values[0]
            embedding = np.array(values[1:], dtype=np.float32)
            embeddings[entity_id] = embedding
    return embeddings

# 提取新闻的WikidataId
def extract_wikidata_ids(entities_str):
    if not isinstance(entities_str, str):
        return []
    entities = eval(entities_str) if entities_str else []
    wikidata_ids = [entity['WikidataId'] for entity in entities]
    return wikidata_ids

# 获取用户点击历史的嵌入序列
def get_user_embedding_sequence(history, news_dict, entity_embeddings, max_length=512):
    if pd.isna(history):
        return np.zeros((max_length, 108), dtype=np.float32)  # 假设embedding的维度是200

    embeddings = []
    for news_id in history.split():
        if news_id in news_dict:
            news_info = news_dict[news_id]
            # print(news_id, news_info, news_info['title_entities'], type(news_info['title_entities']))
            title_wikidata_ids = extract_wikidata_ids(news_info['title_entities'])
            abstract_wikidata_ids = extract_wikidata_ids(news_info['abstract_entities'])
            wikidata_ids = title_wikidata_ids + abstract_wikidata_ids
            # print(wikidata_ids)
            embedding = []
            for wikidata_id in wikidata_ids:
                if wikidata_id in entity_embeddings:
                    embedding.append(entity_embeddings[wikidata_id])
                # else:
                    # embedding.append(np.zeros(100, dtype = np.float32))
            if len(embedding) == 0:
                continue
            if len(embeddings) >= max_length:
                break
            embedding = np.pad(np.mean(embedding, axis = 0), (0, 8), mode='constant', constant_values=0)
            if embedding.shape != (108, ):
                print(embedding.shape)
            embeddings.append(embedding)
            # print(embedding)
        if len(embeddings) >= max_length:
            break
    
    # 如果嵌入数量不足max_length，进行padding
    if len(embeddings) < max_length:
        embeddings += [np.zeros(108, dtype=np.float32)] * (max_length - len(embeddings))
    if np.array(embeddings[:max_length]).shape != (512, 108):
        print("FUCK", np.array(embeddings[:max_length]))
    return np.array(embeddings[:max_length])

# 加载数据
behaviors = load_behaviors('MINDsmall_train/behaviors.tsv')
news = load_news('MINDsmall_train/news.tsv')
entity_embeddings = load_embeddings('MINDsmall_train/entity_embedding.vec')

# 将news数据转换为字典
news_dict = news.set_index('news_id').T.to_dict()

cnt = 0
# 处理所有用户的点击历史
user_history_embeddings = []
for index, row in behaviors.iterrows():
    user_embedding_sequence = get_user_embedding_sequence(row['history'], news_dict, entity_embeddings)
    if user_embedding_sequence.shape != (512, 108):
        continue
    # print(user_embedding_sequence.shape)
    cnt += 1
    user_history_embeddings.append(user_embedding_sequence)
    if cnt >= 5120:
        break
    if cnt % 1000 == 0:
        print(cnt)

# 将用户历史嵌入转化为张量
user_history_embeddings = torch.tensor(user_history_embeddings)

print(user_history_embeddings.shape)  # 应为 (U, maxlen, 100)

# 保存处理后的用户嵌入
torch.save(user_history_embeddings, 'user_history_embedding.pt')
