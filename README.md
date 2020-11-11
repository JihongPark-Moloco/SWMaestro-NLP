# NLP Repository
![영산 컨텐츠 키워드 추출 기법](/uploads/4707fe5fae802bf5bd40cbff71439477/image.png)  
MUNA 팀의 CreateTrend 프로젝트에서 자연어처리 관련 동작을 수집한 레포입니다.  
유튜브 영상의 컨텐츠 키워드를 추출하는 기능을 수행합니다.

## 적용 기술 설명
### Built With
현 프로젝트는 다음의 주요 프레임워크를 통해 개발되었습니다.
* [KR-WorkRank](https://github.com/lovit/KR-WordRank)
* [Yake](https://github.com/LIAAD/yake)
* [FastText](https://github.com/facebookresearch/fastText)

### PipeLine
![컨텐츠_키워드_추출_프로세스](/uploads/2b72ee89675aa8db244a5e6146949050/컨텐츠_키워드_추출_프로세스.png)
1. GCP Vision API를 이용해 썸네일에서 글귀를 분리해냅니다.
2. 제목 설명 댓글에서 KR-WordRank와 YAKE를 이용해 키워드 후보를 추출합니다.
3. 2번에서 추출한 키워드 후보와 영상 태그를 모두 FastText로 임베딩합니다.  
  이후 HDBScan을 이용해 키워드를 압축하고 DB에 저장합니다.

### Why FastText
![유튜브 비정형 댓글들](/uploads/e65a01e5e84cf282563783a6d51dc782/화면_캡처_2020-11-08_144803.png)  
유튜브 커뮤티니 특성상 비문법적이고 비정형적인 은어들 위주의 댓글이 많이 분포합니다.  
substring의 빈도수를 기반으로 키워드를 추출하는 알고리즘 특성상 데이터셋이 부족할경우 조사를 함께 키워드로 인식합니다.  
조사를 제거하기 위해 형태소 분석기를 사용할 경우 신조어를 모두 인식할 수 없는 문제가 존재합니다.  
N-Gram 방식의 임베딩은 단어를 여러 단어로 나누어서 인식하고 임베딩하기에 노이즈에 강인하며 신조어의 학습에도 오랜 시간이 걸리지 않습니다.  
따라서 저희는 FastText를 이용해 키워드를 임베딩합니다.

## 컨텐츠 키워드 추출
### GCP Vision API OCR
![GCP OCR 예시](/uploads/c1b86712d9c9c1b922fccd12e73bc278/화면_캡처_2020-11-08_145424.png)  
![OCR 입력 예시 및 성능 검증](/uploads/e5f0dd505b9b0bb528325e2d0ace47cf/화면_캡처_2020-11-08_145801.png)  
GCP의 Vision API중 OCR 항목을 이용해 유튜브 썸네일에 포함된 글귀를 추출합니다.  
이때 API 비용을 최적화 하기 위해 Document를 분석하였고 해상도와 용량 제한만이 존재함을 확인했습니다.  
이를 이용해 다수의 썸네일을 이어붙여 하나의 입력으로 활용하고 이를 인식된 키워드를 좌표로 다시 썸네일별로 나누어 DB에 저장합니다.  
썸네일을 이어 붙였을때 발생할 수 있는 OCR 성능 저하를 판단하기 위해 각기 썸네일을 인식한 전체 썸네일 키워드의 개수와  
하나의 사진으로 이어붙여 인식했을때 도출되는 키워드 개수를 비교해 그 수가 동일함을 확인했습니다.  
이를 통해 GCP OCR API의 비용을 최적화했습니다.  

### 후보 키워드 추출
![후보 키워드 추출](/uploads/e3c902c2a651cb896eb34c79aa7534c5/화면_캡처_2020-11-08_150236.png)  
영상 제목, 설명, 댓글, 썸네일 글귀로부터 후보 키워드를 추출합니다.  
추출에는 한글은 KR-WordRank, 영어는 YAKE를 사용합니다.  


### 키워드 압축
![유사 키워드 예시](/uploads/b2edc2b929c36c755a3f1cc4c4a89635/화면_캡처_2020-11-08_150527.png)  
![키워드 압축 예시](/uploads/175fe6857aece2c5d87a0ee5f1f9d5f6/image.png)  
추출한 후보 키워드와 영상 태그를 결합하기에 중복되는 태그와 축약형, 영어 등의 유사한 키워드가 섞여서 결과로 나오게됩니다.  
유사 키워드를 하나의 단어로 압축하기위해 FastText로 임베딩한 값을 HDBScan으로 클러스터링하며 군집화된 키워드의 경우  
해당 군집의 대표 키워드를 선정해 키워드를 압축합니다.

## 핵심 소스 설명
- app.py  
  RabbitMQ를 이용한 데이터 송수신으로 서버에서 오는 요청에 대해 키워드를 추출해 Response하게 됩니다.  
  
- fasttext_youtube.py  
  DB에서 저장된 데이터를 이용해 FastText 모델을 유튜브 데이터셋에 맞게 학습시키고 키워드를 추출합니다.  
  또한 키워드 유사도 분석을 통해 유사 채널을 분석해 나타냅니다.
  
- gcp_ocr.py  
  GCP의 Vision API 중 OCR 항목을 이용해 영상 썸네일로부터 글귀를 추출합니다.
  썸네일을 이어붙여 하나의 입력으로 처리하고 이후 리턴되는 키워드를 좌표로 나누어 각 썸네일 항목으로 나누어 DB에 저장합니다.  
  
- keyword_extractor.py
  KR-WordRank와 YAKE를 이용해 입력된 Stirng List로 부터 키워드를 추출해 반환합니댜.

- delete_stop_words_from_db.py  
  불용어로 지정된 키워드를 DB에서 제거하는 역할을 수행합니다.

## 활용 기술들
### KR-WordRank
WordRank 는 띄어쓰기가 없는 중국어와 일본어에서 graph ranking 알고리즘을 이용하여 단어를 추출하기 위해 제안된 방법입니다.
Ranks 는 substring 의 단어 가능 점수이며, 이를 이용하여 unsupervised word segmentation 을 수행하였습니다.
WordRank 는 substring graph 를 만든 뒤, graph ranking 알고리즘을 학습합니다.
Substring graph 는 아래 그림의 (a), (b) 처럼 구성됩니다.
먼저 문장에서 띄어쓰기가 포함되지 않은 모든 substring 의 빈도수를 계산합니다.
이때 빈도수가 같으면서 짧은 substring 이 긴 substring 에 포함된다면 이를 제거합니다.
아래 그림에서 ‘seet’ 의 빈도수가 2 이고, ‘seeth’ 의 빈도수가 2 이기 때문에 ‘seet’ 는 graph node 후보에서 제외됩니다.
두번째 단계는 모든 substring nodes 에 대하여 links 를 구성합니다.
‘that’ 옆에 ‘see’와 ‘dog’ 이 있었으므로 두 마디를 연결합니다.
왼쪽에 위치한 subsrting 과 오른쪽에 위치한 subsrting 의 edge 는 서로 다른 종류로 표시합니다.
이때, ‘do’ 역시 ‘that’의 오른쪽에 등장하였으므로 링크를 추가합니다.
이렇게 구성된 subsrting graph 에 HITS 알고리즘을 적용하여 각 subsrting 의 ranking 을 계산합니다.
![image](https://13.125.91.162/swmaestro/muna-1/raw/master/images/graph_wordrank_algorithm.png)  

유튜브의 경우에는 10대~30대 사이의 젊은 층의 점유율이 매우 높기에 은어와 신조어등에 대해 매우 민감하게 반응합니다.
키워드 추출에서 지도 학습으로 접근할 경우 새롭게 파생되고 변형되어지는 모든 키워드들에 대응하기란 불가능합니다.
그렇기에 비지도 학습 기반의 WordRank를 이용해 단어의 반복해서 나타나는 단어의 빈도수를 파악해 키워드를 추출합니다.
WordRank를 이용할 시 통계에 기반하여 키워드를 추출하기에 사전 데이터 학습이 필요하지 않으며 새롭게 생겨나는 단어에 강인하
사용자가 실수로 발생시키는 오탈자등은 희석되어 전체 키워드 분석에서 제외되므로 키워드 추출 알고리즘으로 매우 적합합니다.

### YAKE
Unsupervised 방식의 텍스트 기반 자동 키워드 추출.
![YAKE](/uploads/f7be42ab813beb34439f227eea384677/image.png)
YAKE는 Unsupervised 방식의 경량 키워드 추출기로
단일 문서에서 추출한 텍스트 통계를 바탕으로 텍스트의 가장 중요한 키워드를 선택한다.
통계 방식에 기반하기 때문에 특정 문석 집합에 대해 훈련될 필요가 없으며, 사전, 텍스트 길이, 언어와 도메인에 따라 달라지지 않습니다.
단일 문서에서 추출한 텍스트 통계 기능을 바탕으로 텍스트의 가장 중요한 키워드를 선택한다.
우리 시스템은 특정 문서 집합에 대해 훈련될 필요가 없으며, 사전, 외부 장식물, 텍스트 크기, 언어 또는 도메인에 따라 달라지지 않는다.

### FastText
![fasttext_vector_field](/uploads/e9833ac68a57ac446e48808d024b4b37/fasttext_vector_field.png)
FastText는 Facebook AI 연구소에서 제작한 N-gram 방식의 단어 임베딩 및 텍스트 분류 학습 라이브러리입니다.
FastText를 통해 영상으로부터 추출한 키워드를 학습시켜 단어를 fasttext 공간에 나타내고 채널별로 빈도수가 높게 나타나는
단어들의 임베딩 값을 통해 해당 채널의 좌푯값을 fasttext 공간 위에 특정합니다.
모든 채널을 fasttext 공간위의 특정짓고 나면 이후 cosine similarity 방식을 통해 유사한 채널을 계산할 수 있습니다.

## Authors
- **박지홍(qkrwlghddlek@naver.com)**