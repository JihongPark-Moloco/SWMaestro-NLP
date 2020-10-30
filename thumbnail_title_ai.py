from elasticsearch import Elasticsearch

es = Elasticsearch("http://www.thecreatetrend.com:9200/")  # 환경에 맞게 바꿀 것
# es.info()
# res = es.search(index='videos', body={
#     "query": {
#         "bool": {
#             "must": [
#                 {"term": {"videokeywordnews.keyword": "asmr"}},
#                 {"term": {"videokeywordnews.keyword": "수면"}},
#                 {"match": {"videokeywordnews.keyword": "팅글 하쁠리"}}
#             ],
#             "must_not": [
#                 {"terms": {"videokeywordnews.keyword": ["입", "나무"]}}
#             ]
#         }
#     }
# })
# advanced_do("팅글 하쁠리", "asmr 수면", "입 나무")

# 예시 keyword_string: "롤 탑 강좌"
def simple_do(keyword_string):
    res = es.search(index='videos', body={
        "query": {
            "match": {"videokeywordnews.keyword": keyword_string}
        }
    })

    idxs = [row['_source']['idx'] for row in res['hits']['hits']]
    return idxs


# 예시 keyword_string: "롤 탑 강좌", must_keyword: "다리우스 티모", must_not_keyword: "정글 미드"
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
