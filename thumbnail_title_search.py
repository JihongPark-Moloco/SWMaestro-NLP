"""
ElasticSearch에 키워드 조회 기능을 구현하기 위한 함수구현입니다.
"""

from elasticsearch import Elasticsearch

es = Elasticsearch("http://www.thecreatetrend.com:9200/")

# 예시 keyword_string: "롤 탑 강좌"
# 롤, 탑, 강좌의 keyword를 포함하는 score가 높은 영상들
def simple_do(keyword_string):
    res = es.search(index='videos', body={
        "query": {
            "match": {"videokeywordnews.keyword": keyword_string}
        }
    })

    idxs = [row['_source']['idx'] for row in res['hits']['hits']]
    return idxs


# 예시 keyword_string: "롤 탑 강좌", must_keyword: "다리우스 티모", must_not_keyword: "정글 미드"
# 다리우스와 티모가 들어가고 (and) 정글이나 미드가 들어가지 않는(or) 영상 중
# 롤, 탑, 강좌의 keyword를 포함하는 score가 높은 영상들
def advanced_do(keyword_string, must_keyword, must_not_keyword):
    body = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"videokeywordnews.keyword": keyword_string}}
                ],
                "must_not": []
            }
        }
    }

    for keyword in must_keyword.split(" "):
        body["query"]["bool"]["must"].append({"term": {"videokeywordnews.keyword": keyword}})

    body["query"]["bool"]["must_not"].append({"terms": {"videokeywordnews.keyword": must_not_keyword.split(" ")}})

    res = es.search(index='videos', body=body)
    idxs = [row['_source']['idx'] for row in res['hits']['hits']]
    return idxs
