from sentiment_analysis import muna_kobert
import torch
import pandas as pd

device = torch.device("cuda:0")
kobert = muna_kobert.KoBERT(device)

### train kobert model
# kobert.make_data_loader("ratings_train.txt")
# kobert.train()

### evaluate kobert model
kobert.make_data_loader("custom_comment.txt", shuffle=False, no_label=True)
# kobert.download_model("trained")
kobert.load_model("trained")
kobert.predict(result_file_name="result2.csv")
