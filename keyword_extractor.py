import bs4
import pandas as pd
import re
import psycopg2 as pg2
import numpy as np
import pandas as pd

from krwordrank.word import KRWordRank


# 댓글 전처리 함수
def pre(text):
    text = text.strip().lower()
    # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z:/_.\'\",~!?`#%^&*{}()]+', '', text)
    text = re.sub('http.*', '', text)
    text = re.sub('[0-9]{1,3}:[0-9]{1,2}', ' ', text)
    # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z!?.,\'\"~`#%^&*(){}]+', ' ', text)
    text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z#]', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub(' {2,}', ' ', text)
    return text.strip()


# 댓글 입력시 키워드를 출력한다
def do_wr_keyword(video_name, video_description, comments):
    min_count = 2  # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length = 10  # 단어의 최대 길이
    wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length, verbose=True)

    beta = 0.85  # PageRank의 decaying factor beta
    max_iter = 10
    keywords, rank, graph = wordrank_extractor.extract([video_name, video_description] + comments, beta, max_iter)

    exact_keys = re.findall('#[ㄱ-ㅣ가-힣a-zA-Z0-9]+', video_description)

    # main_sentence = ''.join(plus_comments)
    print("#### wordrank, 영상 설명에 포함된 키워드 ####")
    print(exact_keys)

    print("#### wordrank, 제목 및 설명 포함 키워드 목록 ####")
    for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        if word in video_name or word in video_description:
            print('%8s:\t%.4f' % (word, r))

    print("#### wordrank, 전체 키워드 목록 ####")

    for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:30]:
        print('%8s:\t%.4f' % (word, r))


conn = pg2.connect(database="createtrend", user="muna", password="muna112358!", host="222.112.206.190",
                   port="5432")
cur = conn.cursor()

cur.execute(f'SELECT idx, channel_id FROM channel;')
channel_id_list = cur.fetchall()

cur.execute(f'SELECT idx, video_name, video_description FROM video WHERE idx=63041;')
video_idx, video_name, video_description = cur.fetchall()[0]

cur.execute(f"SELECT comment_content FROM comment WHERE video_idx={video_idx};")
comments = [pre(c[0]) for c in cur.fetchall()]

do_wr_keyword(pre(video_name), pre(video_description), comments)

conn.close()
