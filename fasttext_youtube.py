"""
FastText 모델을 유튜브 데이터를 이용해 학습시킵니다.
FastText로 키워드를 임베딩하고 각 채널의 평균 좌표를 구해
채널간 유사도 분석을 실시합니다.
"""

import collections
import csv
import re

import hdbscan
import numpy as np
import psycopg2 as pg2
from gensim import models
from scipy.spatial import distance

## 추가 학습 시키는 소스
# IP = IP
# video_keyword = pd.read_csv(r'D:\createtrend_public_video_keyword.csv')
ko_model = models.fasttext.load_facebook_model(r"D:\share\wiki.ko.bin")


def pre(text):
    text = text.strip().lower()
    # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z:/_.\'\",~!?`#%^&*{}()]+', '', text)
    text = re.sub("http.*", "", text)
    text = re.sub("[0-9]{1,3}:[0-9]{1,2}", " ", text)
    # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z!?.,\'\"~`#%^&*(){}]+', ' ', text)
    text = re.sub("[^ 0-9ㄱ-ㅣ가-힣a-z]", " ", text)
    text = re.sub("\n", " ", text)
    text = re.sub(" {2,}", " ", text)
    return text.strip()


new_data = []

# 추가 학습을 위해 DB로 부터 추출한 영상 정보
f = open(r"D:\createtrend_public_video.csv", "r", encoding="utf-8")
rdr = csv.reader(f)
for line in rdr:
    new_data.append(pre(line[1]).split())
    new_data.append(pre(line[2]).split())

# 유튜브 데이터로 추가 학습
ko_model.build_vocab(new_data, update=True)
ko_model.train(new_data, total_examples=len(new_data), epochs=ko_model.epochs)

# 학습된 모델 저장
ko_model.save("new_model")

############ model 구현 소스 ###########
ko_model = models.fasttext.FastText.load("0908_model")

conn = pg2.connect(
    database="createtrend", user="muna", password="muna112358!", host=IP, port="5432"
)
conn.autocommit = False
cur = conn.cursor()
cur.execute(f"""SELECT idx FROM channel WHERE status = true""")
channel_list = cur.fetchall()

channel_vector = []
index_to_name = dict()
index_to_idx = dict()
idx_to_index = dict()

index = 0

# 임베딩된 키워드 벡터 값으로부터 FastText 공간상의 채널의 평균 지점을 계산합니다.
for channel_idxx in channel_list:
    channel_idx = channel_idxx[0]

    cur.execute(f"""SELECT channel_name FROM channel WHERE idx = {channel_idx};""")
    channel_name = cur.fetchall()[0][0]

    cur.execute(
        f"""SELECT * FROM (
           SELECT keyword, count(keyword) as cc
           FROM video_keyword
           WHERE video_idx IN (SELECT idx FROM video WHERE channel_idx = {channel_idx})
           GROUP BY keyword
           ORDER BY cc DESC
    ) as res WHERE res.cc >=2 LIMIT 100;"""
    )

    res = cur.fetchall()
    all = np.zeros(300)
    total_weight = 0

    for keyword, weight in res:
        all += ko_model.wv.get_vector(keyword) * weight
        total_weight += weight

    if total_weight == 0:
        continue

    all /= total_weight

    index_to_name[index] = channel_name
    index_to_idx[index] = channel_idx
    idx_to_index[channel_idx] = index
    channel_vector.append(all)

    index += 1

cnp = np.array(channel_vector)

# 모델 저장 및 불러오기
# with open("0908_data.pickle", "wb") as f:
#     pickle.dump([cnp, index_to_name, index_to_idx, idx_to_index], f)
#
# with open("0908_data.pickle", "rb") as f:
#     cnp, index_to_name, index_to_idx, idx_to_index = pickle.load(f)

# 클러스터링 수행
clusterer = hdbscan.HDBSCAN()
cluster_res = clusterer.fit(cnp)
labels = clusterer.labels_
res = collections.Counter(labels).items()

# 클러스터링 결과 확인
# {k: v for k, v in sorted(dict(res).items(), key=lambda item: item[1], reverse=True)}

# 거리 계산 수행
clustered_index = [i for i, l in enumerate(labels) if l == 12]
mean_vector = np.mean([cnp[i] for i in clustered_index], axis=0)
distances = distance.cdist([mean_vector], cnp, "cosine")[0]

names = [index_to_name[index] for index in clustered_index]
idxs = [index_to_idx[index] for index in clustered_index]

num_channels = 100  # 추출할 채널 개수
distances = distance.cdist([cnp[idx_to_index[12]]], cnp, "cosine")[0]

# 거리가 가까운 순으로 목록을 가져옵니다.
ind = np.argpartition(distances, num_channels)[:num_channels]
for i in np.argpartition(distances, range(num_channels))[:num_channels]:
    print(index_to_idx[i], index_to_name[i], distances[i])

# BallTree를 활용한 사전 학습 트리 생성
# tree = neighbors.BallTree(cnp)
# tree.query([cnp[499]], 10)
# idx_to_index[2888]
# index_to_name[idx_to_index[254]]
# for i in range(len(temp)):
#     res = []
#     for j in range(len(temp)):
#         res.append(1 - spatial.distance.cosine(temp[i], temp[j]))
#     total.append(res)
#
# for line in total:
#     line = [str(e) for e in line]
#     print("\t".join(line))
