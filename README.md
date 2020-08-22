<!-- ABOUT THE PROJECT -->
## About The Project
영상의 제목, 설명, 댓글을 통해 해당 영상의 키워드를 추출합니다.  
추출된 키워드를 통해 FastText 공간에서 채널들간의 거리를 계산해 유사 채널들을 찾아냅니다.  


### Built With
현 프로젝트는 다음의 주요 프레임워크를 통해 개발되었습니다.
* [KR-WorkRank](https://github.com/lovit/KR-WordRank)
* [Yake](https://github.com/LIAAD/yake)
* [FastText](https://github.com/facebookresearch/fastText)

### PipeLine
![image](https://13.125.91.162/swmaestro/muna-1/raw/master/images/NLP_pipeline.png)  
1. 크롤러를 통해 수집된 데이터에서 한글과 영어를 분리해 각기 다른 알고리즘으로 키워드를 추출합니다.
2. 사전에 정의된 StopWord를 걸러냅니다.
3. FastText를 통해 유사채널을 분석합니다.

### Algorithm
#### Keword Extraction
##### KR-WordRank
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
WordRank를 이용할 시 통계에 기반하여 키워드를 추출하기에 사전 데이터 학습이 필요하지 않으며 새롭게 생겨나는 단어에 강인하고 
사용자가 실수로 발생시키는 오탈자등은 희석되어 전체 키워드 분석에서 제외되므로 키워드 추출 알고리즘으로 매우 적합합니다.  
##### YAKE
Unsupervised 방식의 텍스트 기반 자동 키워드 추출.

YAKE는 Unsupervised 방식의 경량 키워드 추출기로
단일 문서에서 추출한 텍스트 통계를 바탕으로 텍스트의 가장 중요한 키워드를 선택한다.
통계 방식에 기반하기 때문에 특정 문석 집합에 대해 훈련될 필요가 없으며, 사전, 텍스트 길이, 언어와 도메인에 따라 달라지지 않습니다.
단일 문서에서 추출한 텍스트 통계 기능을 바탕으로 텍스트의 가장 중요한 키워드를 선택한다. 
우리 시스템은 특정 문서 집합에 대해 훈련될 필요가 없으며, 사전, 외부 장식물, 텍스트 크기, 언어 또는 도메인에 따라 달라지지 않는다. 
#### Analogous Channel Analyzer
![image](https://13.125.91.162/swmaestro/muna-1/raw/master/images/fasttext_vector_field.png)   
FastText는 Facebook AI 연구소에서 제작한 N-gram 방식의 단어 임베딩 및 텍스트 분류 학습 라이브러리입니다.  
FastText를 통해 영상으로부터 추출한 키워드를 학습시켜 단어를 fasttext 공간에 나타내고 채널별로 빈도수가 높게 나타나는
단어들의 임베딩 값을 통해 해당 채널의 좌푯값을 fasttext 공간 위에 특정합니다.  
모든 채널을 fasttext 공간위의 특정짓고 나면 이후 cosine similarity 방식을 통해 유사한 채널을 계산할 수 있습니다.