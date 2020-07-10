# 댓글 감성 분석 레포지토리
KoBERT를 활용해 유튜브 영상 댓글의 감성 정보를 추출합니다.  
  

## Models
#### KoBERT
SKTBrain에서 제작한  
구글 BERT base multilingual cased의 한국어 버전
https://github.com/SKTBrain/KoBERT#tokenizer


### Why KoBERT?
감성 분석을 위한 네트워크로 다음의 세 모델을 고려할 수 있었습니다.
* XLNET  
  XLNet은 GPT로 대표되는 auto-regressive(AR) 모델과 BERT로 대표되는 auto-encoder(AE) 모델의 장점만을 합한 generalized AR pretraining model입니다.  
  하지만 한국어 감성분석에 있어서는 KoBERT의 성능 **90.1%** 에 못 미칩니다.  
  ![image](https://13.125.91.162/swmaestro/muna-1/raw/Sentiment_Analysis/images/XLNET_accuracy.png)  
  https://github.com/yeontaek/XLNET-Korean-Model
* RoBERTa
  Facebook AI와 UW에서 공동 발표한 RoBERTa는 Robustly Optimized BERT Pretraining Approach의 약자로 이름 그대로 BERT의 하이퍼파라미터 및 학습 데이터 사이즈 등을 조절함으로써 기존의 BERT의 성능뿐만 아니라 post-BERT의 성능과 버금가거나 훨씬 더 나을 수 있도록 제작한 model.
  BERT의 12 Layer에 두배에 달하는 24 Layer로 구성되어있지만 감성 분석에서는 기존의 BERT 모데로가 성능적으로 큰 차이가 존재하지 않는다.  
  ![image](https://13.125.91.162/swmaestro/muna-1/raw/Sentiment_Analysis/images/RoBERTa_accuracy.png)  
  _민진우, 나승훈, 신종훈, 김영길 (2019). RoBERTa를 이용한 한국어 자연어처리: 개체명 인식, 감성분석, 의존파싱. 한국정보과학회 학술발표논문집, 407-409_
  
## Files
* **muna_kobert.py**  
  Create Trend 프로젝트에 맞춰 최적화 시킨 KoBERT 모델, 또한 활요을 위해 패키징 해놓은 소스  
  
* muna_kobert_test.py  
  muna_kobert 패키지의 사용예시