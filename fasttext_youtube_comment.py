import re

import psycopg2 as pg2
from gensim import models
import tqdm


def pre(text):
    text = text.strip().lower()
    # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z:/_.\'\",~!?`#%^&*{}()]+', '', text)
    text = re.sub('http.*', '', text)
    text = re.sub('[0-9]{1,3}:[0-9]{1,2}', ' ', text)
    # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z!?.,\'\"~`#%^&*(){}]+', ' ', text)
    text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z]', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub(' {2,}', ' ', text)
    return text.strip()


# f = open(r'D:\createtrend_public_video.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# for line in rdr:
#     line


IP = "ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com"

ko_model = models.fasttext.load_facebook_model(r'D:\share\wiki.ko.bin')

new_data = []
# sample_video_idx_list = [20200, 20387, 20421, 20441, 20506, 20555, 41945, 58156, 83182, 134854, 134858, 134858, 134856,
#                          134856, 195421, 195572, 196810, 196878, 196898, 197014, 197031, 197090, 200181, 214209, 214186,
#                          233272, 244477, 244490, 247600, 247481, 248487, 248487, 249239, 249914, 249921, 249935, 249936,
#                          251524, 251525, 251536, 251535, 256875, 272931, 272971, 273012, 273095, 273270, 273347, 273380,
#                          273383, 278960, 297062, 298555, 298556, 298557, 300134, 300136, 300133, 300135, 301266, 303663,
#                          283933]

conn = pg2.connect(database="createtrend", user="muna", password="muna112358!", host=IP,
                   port="5432")
conn.autocommit = False
cur = conn.cursor()

cur.execute(f'SELECT idx FROM video WHERE status = true AND forbidden = false;')
video_list = cur.fetchall()

for video_idx in tqdm.tqdm(video_list):
    video_idx = video_idx[0]
    cur.execute(f'SELECT video_name, video_description FROM video WHERE idx={video_idx};')
    video_name, video_description = cur.fetchall()[0]
    new_data.append(pre(video_name).split())
    new_data.append(pre(video_description).split())
    # cur.execute(f"SELECT comment_content FROM comment WHERE video_idx={video_idx};")
    try:
        cur.execute(
            f"SELECT video_idx, string_agg(keyword, ', ') as cl FROM video_keyword_new WHERE video_idx={video_idx} GROUP BY video_idx")
        keywords = cur.fetchall()[0][1].split(",")
        new_data.append(keywords)
    except:
        pass
    # for comment in comments:
    #     new_data.append(pre(comment[0]).split())

ko_model.build_vocab(new_data, update=True)
ko_model.train(new_data, total_examples=len(new_data), epochs=ko_model.epochs)

ko_model.wv.similar_by_word('임대차3법', 10)
ko_model.save('0908_model')

ko2_model = models.fasttext.FastText.load('617_model')

for w, sim in ko_model.similar_by_word('유튜브', 10):
    print(f'{w}: {sim}')
