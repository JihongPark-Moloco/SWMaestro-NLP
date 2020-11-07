"""
유튜브 댓글을 이용해 FastText모델을 학습시키고 저장합니다.
"""

import re

import psycopg2 as pg2
import tqdm
from gensim import models


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


IP = "ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com"

ko_model = models.fasttext.load_facebook_model(r"D:\share\wiki.ko.bin")

new_data = []

conn = pg2.connect(
    database="createtrend", user="muna", password="muna112358!", host=IP, port="5432"
)
conn.autocommit = False
cur = conn.cursor()

cur.execute(f"SELECT idx FROM video WHERE status = true AND forbidden = false;")
video_list = cur.fetchall()

for video_idx in tqdm.tqdm(video_list):
    video_idx = video_idx[0]
    cur.execute(f"SELECT video_name, video_description FROM video WHERE idx={video_idx};")
    video_name, video_description = cur.fetchall()[0]
    new_data.append(pre(video_name).split())
    new_data.append(pre(video_description).split())
    # cur.execute(f"SELECT comment_content FROM comment WHERE video_idx={video_idx};")
    try:
        cur.execute(
            f"SELECT video_idx, string_agg(keyword, ', ') as cl FROM video_keyword_new WHERE video_idx={video_idx} GROUP BY video_idx"
        )
        keywords = cur.fetchall()[0][1].split(",")
        new_data.append(keywords)
    except:
        pass

ko_model.build_vocab(new_data, update=True)
ko_model.train(new_data, total_examples=len(new_data), epochs=ko_model.epochs)

ko_model.save("0908_model")
