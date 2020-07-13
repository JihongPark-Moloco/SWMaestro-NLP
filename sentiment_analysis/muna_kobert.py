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


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
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

        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

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
                    print(
                        "## epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1,
                                                                              loss.data.cpu().numpy(),
                                                                              train_acc / (batch_id + 1)))
            print("## epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

        print("## Model trained")

    def eval(self):
        print("## Preparing eval...")

        if not self.exist_data_loader:
            print("## No data_loader!!\n## run make_data_loader() first!!")
            return

        self.model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.data_loader)):
            test_acc = 0.0
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            label = label.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            test_acc += self.calc_accuracy(out, label)

        print("## test acc {}".format(test_acc / (batch_id + 1)))

    def predict(self, result_file_name):
        print("## Preparing predict...")

        if not self.exist_data_loader:
            print("## No data_loader!!\n## run make_data_loader() first!!")
            return

        df = pd.DataFrame(columns=["flag"])
        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.data_loader)):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            _, max_indices = torch.max(out, 1)

            df = pd.concat([df, pd.DataFrame(max_indices.data.cpu().numpy(), columns=["flag"])], ignore_index=True)

        df.to_csv(result_file_name)

        print("## result file generated")

    def download_model(self, model_file_name="trained"):
        print("## Model download start ..")

        url = 'https://drive.google.com/u/0/uc?export=download&confirm=WbK4&id=18mlpgwfNznsnviV-NxP7HpsurXKA6Lpv'
        gdown.download(url, model_file_name, quiet=False)

        print("## Model download done")

    def load_model(self, model_file_name):
        print("## Model loading..")

        state = torch.load(model_file_name)
        self.model.load_state_dict(state['model_state_dict'])

        print("## Model loaded..")

    def save_model(self, model_file_name):
        print("## Model saving..")

        torch.save({'model_state_dict': self.model.state_dict()}, model_file_name)

        print("## Model saved")
