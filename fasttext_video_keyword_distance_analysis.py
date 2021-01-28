"""
각 영상들끼리 가장 거리가 가깝고 먼 키워드를 분석해 DB에 저장합니다.
데이터 분석용 소스입니다.
"""

import itertools

import numpy as np
import psycopg2 as pg2
from gensim import models
from tqdm import tqdm

ko_model = models.fasttext.FastText.load("0908_model")

IP = #IP
database = #database
user = #user
password = #password

conn = pg2.connect(
    database=database, user=user, password=password, host=IP, port="5432"
)
cur = conn.cursor()
cur.execute(
    f"""SELECT video_keyword.video_idx, string_agg(keyword, ', ') AS keywords
FROM video_keyword
         JOIN video v on video_keyword.video_idx = v.idx
WHERE v.channel_idx IN
      (12, 148, 156, 60, 17, 26, 94, 168, 55, 1588, 14, 133, 51, 161, 23, 70, 99, 211, 119, 30, 95, 117, 29, 83, 123,
       113, 38, 103, 147, 124, 220, 263, 215, 122, 66, 201, 71, 72, 163, 153, 2, 80, 182, 207, 76, 89, 172, 98, 9, 197)
GROUP BY video_keyword.video_idx;"""
)
res = cur.fetchall()

for video_idx, keywords in tqdm(res):
    min_distance = 1.1
    max_distance = -1.1

    vectors = [ko_model.wv.get_vector(keyword.strip()) for keyword in keywords.split(",")]
    mean = np.mean(vectors)
    var = np.var(vectors)

    for k1, k2 in itertools.permutations(keywords.split(","), 2):
        dd = ko_model.similarity(k1.strip(), k2.strip())

        if dd < min_distance:
            min_distance = dd
            min_distance_k1 = k1
            min_distance_k2 = k2

        if dd > max_distance:
            max_distance = dd

    cur.execute(
        f"""INSERT INTO video_keyword_new_distance (video_idx, min_distance, max_distance, keyword_1, keyword_2, mean, var) VALUES ('{video_idx}', '{min_distance}', '{max_distance}', '{k1}', '{k2}', '{mean}', '{var}')"""
    )

conn.commit()
conn.close()
