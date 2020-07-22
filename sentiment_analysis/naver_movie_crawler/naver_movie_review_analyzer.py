import pandas as pd
import os
import tqdm

file_list = os.listdir("comments")

df = pd.DataFrame(columns=["nickname", "review_text", "score", "date"])

for file in tqdm.tqdm(file_list):
    df = pd.concat([df, pd.read_csv(os.path.join("comments", file))], ignore_index=True)

df = df.loc[df["review_text"].str.len() <= 119]  # 2468388개

len(df.loc[df["score"] == 10])  # 1303183
len(df.loc[df["score"] == 9])  # 205616
len(df.loc[df["score"] == 8])  # 278736
len(df.loc[df["score"] == 7])  # 98256
len(df.loc[df["score"] == 6])  # 118645
len(df.loc[df["score"] == 5])  # 50279
len(df.loc[df["score"] == 4])  # 54287
len(df.loc[df["score"] == 3])  # 22964
len(df.loc[df["score"] == 2])  # 83289
len(df.loc[df["score"] == 1])  # 253133

df_score_4 = df.loc[df["score"] == 4]
df_score_5 = df.loc[df["score"] == 5]
df_score_6 = df.loc[df["score"] == 6]

df_score_4.to_csv("comments_4.csv")
df_score_5.to_csv("comments_5.csv")
df_score_6.to_csv("comments_6.csv")

# df_score_10 = df.loc[df["score"] == 10].sample(n=253133)
# df_score_1 = df.loc[df["score"] == 1]
#
# df_score_1.to_csv("comments_1.csv")
# df_score_10.to_csv("comments_10.csv")
#
# # 1 = positive
# df_score_1_label = pd.DataFrame([0] * len(df_score_1), columns=["label"])
# temp = df_score_1.filter(items=["review_text"])
# modified_df_score_1 = pd.concat([temp.reset_index(), df_score_1_label], axis=1).filter(items=["review_text", "label"])
#
# df_score_10_label = pd.DataFrame([1] * len(df_score_10), columns=["label"])
# temp = df_score_10.filter(items=["review_text"])
# modified_df_score_10 = pd.concat([temp.reset_index(), df_score_10_label], axis=1).filter(items=["review_text", "label"])
#
# df_score_1_train = modified_df_score_1.sample(frac=0.8)
# df_score_1_test = modified_df_score_1.drop(df_score_1_train.index)
#
# df_score_10_train = modified_df_score_10.sample(frac=0.8)
# df_score_10_test = modified_df_score_10.drop(df_score_1_train.index)
#
# df_train = pd.concat([df_score_1_train, df_score_10_train])
# df_test = pd.concat([df_score_1_test, df_score_10_test])
#
# df_train.to_csv("df_train.csv")
# df_test.to_csv("df_test.csv")