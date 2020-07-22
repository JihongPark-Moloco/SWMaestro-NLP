import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
import pandas as pd
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import gdown


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair, no_label=False):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        # np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
        # np.array(segment_ids, dtype='int32')
        # 임베딩된 단어로 나타낸 문장, 네트워크 인풋에 맞추기 위해 패딩인 1이 들어간다
        # input_ids = [2, 2145, 6844, 5859, 6928, 6865, 5770, 7043, 5922, 7095, 7922, 6573, 7100, 6928, 5561, 5691, 6228, 7174, 7850, 5732, 6020, 6855, 6865, 5770, 6150, 5765, 5868, 6659, 6499, 7848, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # 패딩을 제외한 실제 문장 길이
        # valid_length = 31
        # 줄을 나타내는 거지만 여기선 모두 다 0, 한 문장임을 의미한다
        # segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.sentences = [transform([i[sent_idx]]) for i in dataset]

        # mode='labeled'일 경우 해당 라벨을 사용, 다른 모드일 경우 모든 라벨에 0을 집어넣는다
        if no_label:
            self.labels = [np.int32(0)] * len(dataset)
        else:
            self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


# Bert 네트워크 맨 마지막단에 Hidden Layer에서 2개의 Tensor로 이어지는 Classifier Layer 층을 추가한다.
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        # Dropout Rate, 의도적으로 rate 만큼의 네트워크를 비활성화 시켜 네트워크가 골고루 학습되도록 한다
        self.dr_rate = dr_rate
        # hidden size -> num_class 만큼의 텐서 갯수를 가진 Linear 네트워크를 형성
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Dropout 레이트를 가진 Dropout 네트워크를 형성
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        # 배치에서 각 line들의 mask를 valid length 만큼 1로 치환
        # ex) BERT의 입력은 64길이이고 실제 입력되는 문장이 30길이면 마스크는 '1'로 되어진 30개의 요소 + '0'으로 되어진 34개의 요소로 이루어진다.
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        # _에는 64개의 토큰 을 하나씩 마스크를 씌워 분석한 BERT 네트워크의 출력이 리턴되며 pooler에는 맨 처음 CLS 토큰을 마스크를 씌운
        # 출력만이 나타난다.
        # ex) batch가 32, max_length=64, hidden_size=768일때
        # _ = [ 32 x 64 x 768 ] size
        # pooler = [ 32 x 768 ] size
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))

        # 위에서 형성한 dropout에 BERT의 출력 (pooler)를 넣어준다
        if self.dr_rate:
            out = self.dropout(pooler)

        # dropout의 결과 out을 맨 마지막 Linear 네트워크에 넣어서 그 출력을 리턴한다.
        return self.classifier(out)


class KoBERT:
    def __init__(self, device, num_classes=2):
        self.exist_data_loader = False
        print("## Making KoBERT Model...")

        bertmodel, self.vocab = get_pytorch_kobert_model()

        self.device = device
        self.model = BERTClassifier(bertmodel, dr_rate=0.5, num_classes=num_classes).to(device)

        print("## Maked KoBERT Model")

    def make_data_loader(self, data_file, vocab=None, num_workers=0, shuffle=True, batch_size=32, max_len=64,
                         no_label=False):
        print("## Making data_loader...")

        if not vocab:
            vocab = self.vocab

        # BERT에서 사용하는 SentencePiece 토크나이저를 생성한다.
        # 해당 SP는 SKT 에서 사전에 학습시킨 vocab으로 만들어진다.
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        # no_label=1일 경우 데이터셋의 라벨을 쓰지 않고 전부 0으로 라벨을 넣어준다. 라벨링이 되어있지 않은 데이터(테스트용)을 위해 제작되었다.
        if no_label:
            dataset = nlp.data.TSVDataset(data_file, field_indices=[1], num_discard_samples=1)
        else:
            dataset = nlp.data.TSVDataset(data_file, field_indices=[1, 2], num_discard_samples=1)

        data = BERTDataset(dataset, 0, 1, tok, max_len, True, False, no_label=no_label)
        self.data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                                       pin_memory=True,
                                                       num_workers=num_workers)
        self.exist_data_loader = True

        print("## Maked data_loader...")

    # Classifier 의 출력인 맨 마지막 두 텐서의 값을 비교해 0 또는 1의 라벨을 붙이는 함수
    def calc_accuracy(self, X, Y):
        max_vals, max_indices = torch.max(X, 1)
        acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
        return acc

    def train(self, warmup_ratio=0.1, num_epoch=5, max_grad_norm=1, log_interval=200, learning_rate=5e-5):
        print("## Preparing training...")

        if not self.exist_data_loader:
            print("## No data_loader!!\n## run make_data_loader() first!!")
            return

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        t_total = len(self.data_loader) * num_epoch
        warmup_step = int(t_total * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        self.model.train()

        for e in range(num_epoch):
            train_acc = 0.0

            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.data_loader)):
                optimizer.zero_grad()
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                label = label.long().to(self.device)
                out = self.model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_acc += self.calc_accuracy(out, label)
                if batch_id % log_interval == 0:
                    print("## epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1,
                                                                                loss.data.cpu().numpy(),
                                                                                train_acc / (batch_id + 1)))

            print("## epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

            # 각 에폭마다 모델을 저장한다.
            self.save_model(f"trained_model_{e + 1}")
            print("## epoch {} model saved ".format(e + 1, ))

        print("## Model trained")

    # 라벨링 된 데이터를 넣어서 BERT의 예측값과 비교, 일치율을 나타내어준다.
    def eval(self):
        print("## Preparing eval...")
        if not self.exist_data_loader:
            print("## No data_loader!!\n## run make_data_loader() first!!")
            return

        self.model.eval()
        test_acc = 0.0

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.data_loader)):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            label = label.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            test_acc += self.calc_accuracy(out, label)

        print("## test acc {}".format(test_acc / (batch_id + 1)))

    # 라벨링이 되어있지 않는 데이터를 넣어서 각 문장의 라벨을 판단한다.
    def predict(self, result_file_name):
        print("## Preparing predict...")

        if not self.exist_data_loader:
            print("## No data_loader!!\n## run make_data_loader() first!!")
            return

        df = pd.DataFrame(columns=["flag"])
        # 텐서값을 저장하기 위한 임시 데이터프레임
        df2 = pd.DataFrame(columns=["a", "b"])
        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.data_loader)):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            _, max_indices = torch.max(out, 1)

            df2 = pd.concat([df2, pd.DataFrame(out.data.cpu().numpy(), columns=["a", "b"])], ignore_index=True)
            df = pd.concat([df, pd.DataFrame(max_indices.data.cpu().numpy(), columns=["flag"])], ignore_index=True)

        df.to_csv(result_file_name)

        print("## result file generated")
        return df2      # 텐서값을 꺼내기 위해서 df2를 리턴시킨다.

    # 사전 학습 모델
    def download_model(self, model_file_name="trained"):
        print("## Model download start ..")

        url = 'https://drive.google.com/u/0/uc?export=download&confirm=WbK4&id=18mlpgwfNznsnviV-NxP7HpsurXKA6Lpv'
        gdown.download(url, model_file_name, quiet=False)

        print("## Model download done")

    # 입력한 파일의 모델을 불러온다.
    def load_model(self, model_file_name):
        print("## Model loading..")

        state = torch.load(model_file_name)
        self.model.load_state_dict(state['model_state_dict'])

        print("## Model loaded..")

    # 현재 모델을 저장한다.
    def save_model(self, model_file_name):
        print("## Model saving..")

        torch.save({'model_state_dict': self.model.state_dict()}, model_file_name)

        print("## Model saved")
