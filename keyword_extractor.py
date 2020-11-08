"""
주어진 텍스르 리스트로부터 키워드 후보들을 추출해 DB에 저장합니다.
"""

import re
import traceback

import psycopg2 as pg2
import yake
from krwordrank.word import KRWordRank

IP = "ec2-13-124-107-195.ap-northeast-2.compute.amazonaws.com"


class keyword_extractor:
    do_sql = False

    # 댓글 전처리 함수
    def pre(self, text):
        text = text.strip().lower()
        # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z:/_.\'\",~!?`#%^&*{}()]+', '', text)
        text = re.sub("http.*", "", text)
        text = re.sub("[0-9]{1,3}:[0-9]{1,2}", " ", text)
        # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z!?.,\'\"~`#%^&*(){}]+', ' ', text)
        text = re.sub("[^ 0-9ㄱ-ㅣ가-힣a-z]", " ", text)
        text = re.sub("\n", " ", text)
        text = re.sub(" {2,}", " ", text)
        return text.strip()

    def pre_kor(self, text):
        text = text.strip().lower()
        # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z:/_.\'\",~!?`#%^&*{}()]+', '', text)
        text = re.sub("http.*", "", text)
        text = re.sub("[0-9]{1,3}:[0-9]{1,2}", " ", text)
        # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z!?.,\'\"~`#%^&*(){}]+', ' ', text)
        text = re.sub("[^ 0-9ㄱ-ㅣ가-힣]", " ", text)
        text = re.sub("\n", " ", text)
        text = re.sub(" {2,}", " ", text)
        return text.strip()

    def pre_eng(self, text):
        text = text.strip().lower()
        # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z:/_.\'\",~!?`#%^&*{}()]+', '', text)
        text = re.sub("http.*", "", text)
        text = re.sub("[0-9]{1,3}:[0-9]{1,2}", " ", text)
        # text = re.sub('[^ 0-9ㄱ-ㅣ가-힣a-z!?.,\'\"~`#%^&*(){}]+', ' ', text)
        text = re.sub("[^ 0-9a-z]", " ", text)
        text = re.sub("\n", " ", text)
        text = re.sub(" {2,}", " ", text)
        return text.strip()

    # 댓글 입력시 키워드를 출력한다
    def do_wr_keyword(self, video_name, video_description, comments, video_idx):
        min_count = 2  # 단어의 최소 출현 빈도수 (그래프 생성 시)
        max_length = 10  # 단어의 최대 길이
        wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length, verbose=False)

        beta = 0.85  # PageRank의 decaying factor beta
        max_iter = 10
        inputs = [video_name, video_description] + comments
        inputs = [v for v in inputs if v]

        if len(inputs) <= 3:
            print("No Korean")
            return []

        try:
            keywords, rank, graph = wordrank_extractor.extract(inputs, beta, max_iter)
        except ValueError:
            return []

        insert_list = []
        print("#### wordrank, 제목 및 설명 포함 키워드 목록 ####")
        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
            if word in video_name or word in video_description:
                if self.do_sql:
                    if r > 1.0:
                        insert_list.append(f"({video_idx}, '{word[:99]}'),")
                else:
                    print("%8s:\t%.4f" % (word, r))

        print("#### wordrank, 전체 키워드 목록 ####")

        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]:
            if self.do_sql:
                insert_list.append(f"({video_idx}, '{word[:99]}'),")
            else:
                print("%8s:\t%.4f" % (word, r))

        return insert_list

    def do_yake(self, video_name, video_description, comments, video_idx):
        text = " ".join([video_name, video_description, *comments])
        kw_extractor = yake.KeywordExtractor(n=1)
        # print("text:", text)
        if len(text.strip()) == 0:
            return []
        keywords = kw_extractor.extract_keywords(text)
        insert_list = []
        print("#### wordrank, 영어 키워드 목록 ####")
        for word, r in keywords:
            if self.do_sql:
                if r <= 0.1:
                    insert_list.append(f"({video_idx}, '{word[:99]}'),")
            else:
                print("%8s:\t%.4f" % (word, r))
        return insert_list

    def do(self, video_idx):
        try:
            conn = pg2.connect(
                database="createtrend", user="muna", password="muna112358!", host=IP, port="5432",
            )
            conn.autocommit = False
            cur = conn.cursor()
            cur.execute(
                f"SELECT idx, video_name, video_description FROM video WHERE idx={video_idx};"
            )
            video_idx, video_name, video_description = cur.fetchall()[0]
            cur.execute(f"SELECT comment_content FROM comment WHERE video_idx={video_idx};")
            comments = cur.fetchall()

            comments_kor = [self.pre_kor(c[0]) for c in comments]
            comments_eng = [self.pre_eng(c[0]) for c in comments]

            exact_keys = [
                keyword[1:] for keyword in re.findall("#[ㄱ-ㅣ가-힣a-zA-Z0-9]+", video_description)
            ]
            print("#### wordrank, 영상 설명에 포함된 키워드 ####")
            print(exact_keys)

            insert_1_list = self.do_wr_keyword(
                self.pre_kor(video_name), self.pre_kor(video_description), comments_kor, video_idx,
            )
            insert_2_list = self.do_yake(
                self.pre_eng(video_name), self.pre_eng(video_description), comments_eng, video_idx,
            )

            insert_list = insert_1_list + insert_2_list

            if self.do_sql:
                for key in exact_keys:
                    insert_list.append(f"({video_idx}, '{key[:99]}'),")
                    cur.execute(
                        f"INSERT INTO video_keyword (video_idx, keyword) VALUES ({video_idx}, '{key}') ON CONFLICT DO NOTHING"
                    )

            if self.do_sql and len(insert_list) != 0:
                sql = " ".join(
                    [
                        "INSERT INTO video_keyword (video_idx, keyword) VALUES",
                        "".join(insert_list)[:-1],
                        "ON CONFLICT DO NOTHING",
                    ]
                )
                cur.execute(sql)

            if self.do_sql:
                conn.commit()
            conn.close()
            return True
        except:
            print(traceback.format_exc())
            return False
