"""
DB에서 불용어 목록으로 지정된 키워드들을 제거합니다.
"""

import psycopg2 as pg2
import pandas as pd

IP = # IP
database = #database
user = #user
password = #password

conn = pg2.connect(database=database,
                   user=user,
                   password=password,
                   host=IP,
                   port="5432")
conn.autocommit = False
cur = conn.cursor()

csv = pd.read_csv('stops.csv', header=None)
for i, row in csv.iterrows():
    print(row[0])
    sql = f"DELETE FROM video_keyword WHERE keyword = '{row[0]}';"
    cur.execute(sql)

conn.commit()
conn.close()
