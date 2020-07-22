import pandas as pd
import tqdm

with open('sentiment_word.txt', 'r', encoding='UTF-8') as file:
    words = file.readlines()

with open('custom_comment.txt', 'r', encoding='UTF-8') as file:
    sentences = file.readlines()

df = pd.DataFrame(sentences, columns=['sentence'])
df['sentiment'] = 'False'

for index, row in tqdm.tqdm(df.iterrows()):
    for word in words:
        if word.strip() in row['sentence']:
            df.loc[index]['sentiment'] = True
            break
