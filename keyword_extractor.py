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

for idx, channel_id in channel_id_list:
    idx = 530
    cur.execute(f"""
SELECT A.idx, A.video_name, A.video_description, B.views
FROM video A
         LEFT JOIN (SELECT DISTINCT ON (video_idx) video_idx, check_time, views
                    FROM video_views
                    ORDER BY video_idx, check_time DESC) B
                   ON A.idx = B.video_idx
WHERE A.channel_idx = {idx}
  AND A.forbidden = FALSE;
""")
    video_list = pd.DataFrame(cur.fetchall())
    views = video_list[3]
    np.average(views)

cur.execute(f'SELECT idx, video_name, video_description FROM video WHERE idx=63041;')
video_idx, video_name, video_description = cur.fetchall()[0]

cur.execute(f"SELECT comment_content FROM comment WHERE video_idx={video_idx};")
comments = [pre(c[0]) for c in cur.fetchall()]

do_wr_keyword(pre(video_name), pre(video_description), comments)

conn.close()
#
# comments = [pre(''.join(div.find("p").text).strip()) for div in all_divs if
#             pre(''.join(div.find("p").text).strip()) != '']
# ori_comments = [''.join(div.text.split('\n')[1:]).strip() for div in all_divs]
#
# # 제목과 영상 설명 리스트
# plus_data = ["""
# [무한도전] ((무한상사)) 멍X이 = 약간..모자라지만 착한 친구야! (=^_^=) 하루에 400번씩 하는(?) 나쁜 말 고치기 특강 👩‍🏫
# """, """
# 공식홈페이지  http://www.imbc.com/broad/tv/ent/chal...
# 방송시간  SAT 18:30~
# Infinite Challenge(무한도전), EP270, 2011/10/08, MBC TV, Republic of Korea
#
# 배현진 아나운서와 고운말 쓰기 특강
# """]
#
# plus_comments = [pre(text.strip()) for text in plus_data]
# comments += plus_comments
# do_wr_keyword(comments, plus_comments)
