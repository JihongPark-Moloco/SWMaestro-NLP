import bs4
import pandas as pd
import re

from krwordrank.word import KRWordRank

num = 11

with open(f"{num}.html", encoding='UTF-8') as fp:
    soup = bs4.BeautifulSoup(fp, 'html.parser')
    all_divs = soup.find_all("div", {'class': 'Comments'})


# 댓글 전처리 함수
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


comments = [pre(''.join(div.find("p").text).strip()) for div in all_divs if
            pre(''.join(div.find("p").text).strip()) != '']
ori_comments = [''.join(div.text.split('\n')[1:]).strip() for div in all_divs]

# 제목과 영상 설명 리스트
plus_data = ["""
[무한도전] ((무한상사)) 멍X이 = 약간..모자라지만 착한 친구야! (=^_^=) 하루에 400번씩 하는(?) 나쁜 말 고치기 특강 👩‍🏫
""", """
공식홈페이지  http://www.imbc.com/broad/tv/ent/chal...
방송시간  SAT 18:30~
Infinite Challenge(무한도전), EP270, 2011/10/08, MBC TV, Republic of Korea

배현진 아나운서와 고운말 쓰기 특강
"""]


# 댓글 입력시 키워드를 출력한다
def do_wr_keyword(comments, plus_comments):
    min_count = 2  # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length = 10  # 단어의 최대 길이
    wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)

    beta = 0.85  # PageRank의 decaying factor beta
    max_iter = 10
    keywords, rank, graph = wordrank_extractor.extract(comments, beta, max_iter)

    main_sentence = ''.join(plus_comments)

    print("#### wordrank, 제목 및 설명 포함 키워드 목록 ####")
    for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        if word in main_sentence:
            print('%8s:\t%.4f' % (word, r))

    print("#### wordrank, 전체 키워드 목록 ####")

    for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:30]:
        print('%8s:\t%.4f' % (word, r))


plus_comments = [pre(text.strip()) for text in plus_data]
comments += plus_comments
do_wr_keyword(comments, plus_comments)
