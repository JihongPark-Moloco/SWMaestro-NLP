from sentiment_analysis import muna_kobert
import torch
import pandas as pd

device = torch.device("cuda:0")
kobert = muna_kobert.KoBERT(device)

### train kobert model
# kobert.make_data_loader("ratings_train.txt")      # 데이터 불러오기
# kobert.train()                                    # 불러온 데이터로 학습

### evaluate kobert model
# kobert.make_data_loader("ratings_test.txt", shuffle=False, no_label=True) # 데이터를 불러올때 Shuffle 시키지 않고, Labeling이 되어있지 않게 불러온다.
kobert.make_data_loader("custom_comment.txt", shuffle=False, no_label=True)
# # kobert.download_model("trained")                                        # SKT에서 학습한 모델을 다운받아 trained 이름으로 저장한다.
kobert.load_model("trained_model_5")                                        # 해당 파일의 모델을 불러온다
res=kobert.predict("nono")                                                  # 입력한 data의 결과를 'nono'로 저장한다, 텐서값을 리턴받아 저장한다.
res.to_csv("tensor.csv")                                                    # 텐서값을 받아오기 위해 만든 임시 리턴값을 저장한다.
# kobert.predict(result_file_name="result2.csv")
