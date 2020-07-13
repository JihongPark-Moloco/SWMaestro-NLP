import requests
from bs4 import BeautifulSoup
import datetime
import time
import pandas as pd
from torrequest import TorRequest
from multiprocessing import Pool
import multiprocessing
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
}

def get_data(url):
    resp = requests.get(url, headers=headers)
    html = BeautifulSoup(resp.content, 'html.parser', formatter=None)
    score_result = html.find('div', {'class': 'score_result'})
    lis = score_result.findAll('li')
    count = 0
    temp_df = pd.DataFrame(columns=["nickname", "review_text", "score", "date"])

    for li in lis:
        print(count)

        nickname = li.findAll('em')[1].find('span').text
        # nickname = li.findAll('a')[0].find('span').text
        created_at = datetime.datetime.strptime(li.find('dt').findAll('em')[-1].text, "%Y.%m.%d %H:%M")

        if li.find('span', {'id': f'_filtered_ment_{count}'}).find('a'):
            review_text = li.find('span', {'id': f'_filtered_ment_{count}'}).find('a').text
        else:
            review_text = li.find('span', {'id': f'_filtered_ment_{count}'}).text

        score = li.find('em').getText()

        temp_df = temp_df.append(
            {"nickname": nickname.strip(), "review_text": review_text.strip(), "score": score.strip(),
             "date": created_at}, ignore_index=True)
        print(nickname)
        count += 1

    return temp_df


# 다크나이트, 62586
def do_crawling(code):
    # with TorRequest(proxy_port=9050, ctrl_port=9051, password='password') as tr:
    #     response = tr.get('http://ipecho.net/plain')
    #     print('#1', response.text)  # not your IP address
    #     tr.reset_identity()
    #     print('#2', response.text)  # not your IP address
    try:
        process_name = str(multiprocessing.current_process()).split("(")[1].split(",")[0]
        head_url = f'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code={code}&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false'
        resp = requests.get(head_url, headers=headers)
        html = BeautifulSoup(resp.content, 'html.parser')
        result = html.find('strong', {'class': 'total'}).findChildren('em')[0].text
        total_count = int(result.replace(',', ''))

        print(time.asctime(), f"{process_name}", code, total_count)

        df = pd.DataFrame(columns=["nickname", "review_text", "score", "date"])

        for i in range(1, int(total_count / 10) + 1):
            url = head_url + '&page=' + str(i)
            # print('url: "' + url + '" is parsing....')
            df = pd.concat([df, get_data(url)], ignore_index=True)
            time.sleep(1)

        df.to_csv(f"{code}.csv")

    except Exception as e:
        print(str(multiprocessing.current_process()).split("(")[1].split(",")[0])
        print(i)
        print(e)


codes = [
    85579,
    146469,
    142384,
    135874,
    158191,
    161242,
    146506,
    153652,
    144314,
    136872,
    149747,
    134898,
    188909,
    152385,
    150637,
    150198,
    120160,
    140731,
    132626,
    161850,
    82473,
    146517,
    146480,
    127398,
    137890,
    123630,
    146524,
    127382,
    129094,
    149512,
    155716,
    130849,
    117787,
    125473,
    156083,
    143394,
    152341,
    162173,
    116866,
    152625,
    106360,
    127397,
    144927,
    144280,
    160749,
    160399,
    164932,
    152170,
    146485,
    99715,
    137696,
    155256,
    137875,
    152168,
    129095,
    154112,
    125488,
    154668,
    154272,
    149221,
    169349,
    143402,
    110333,
    134838,
    191633,
    116233,
    143473,
    18610,
    152616,
    159037,
    149777,
    137970,
    165722,
    126034,
    153642,
    152396,
    102278,
    137008,
    175318,
    152633,
    159054,
    147092,
    155411,
    144379,
    142699,
    155484,
    118966,
    127374,
    121052,
    137945,
    163844,
    152331,
    154353,
    156675,
    118307,
    76309,
    127346,
    146459,
    155715,
    140010,
    167697,
    136315,
    156464,
    154222,
    154285,
    144330,
    163533,
    137326,
    167638,
    158178,
    151153,
    153687,
    119428,
    167105,
    149236,
    164192,
    151728,
    158180,
    175322,
    159892,
    136990,
    172425,
    140652,
    172454,
    168298,
    169015,
    165748,
    154255,
    149248,
    164115,
    136898,
    162981,
    164106,
    157297,
    152656,
    166092,
    160487,
    174835,
    164139,
    171755,
    154449,
    164101,
    158112,
    162249,
    141206,
    153675,
    165026,
    142739,
    169347,
    125494,
    169643,
    168049,
    160375,
    153651,
    118955,
    109906,
    175727,
    159877,
    172010,
    132996,
    171725,
    155356,
    174805,
    168058,
    158647,
    150688,
    180379,
    173653,
    158601,
    160491,
    155263,
    168050,
    85578,
    152661,
    189368,
    158622,
    158626,
    168023,
    164151,
    166610,
    152156,
    173019,
    159805,
    154226,
    172975,
    146489,
    167787,
    143416,
    154667,
    176354,
    164183,
    152249,
    144266,
    173692,
    156496,
    167602,
    66725,
    168017,
    152680,
    153620,
    167651,
    136900,
    136873,
    163788,
    161967,
    174903,
    187940,
    173123,
    132623,
    167613,
    178526,
    169637,
    177909,
    179482,
    182355,
    101966,
    163608,
    177967,
    183876,
    182205,
    167699,
    177374,
    189053,
    167605,
    167099,
    96951,
    179159,
    181381,
    66728,
    164172,
    164173,
    180351,
    152632,
    174065,
    178544,
    109193,
    175366,
    181414,
    167635,
    181710,
    179158,
    180390,
    174050,
    175316,
    182360,
    159806,
    180374,
    179125,
    187349,
    181698,
    173668,
    186615,
    189046,
    164125,
    97631,
    164907,
    189000,
    183136,
    185933,
    181114,
    167657,
    140649,
    116234,
    167653,
    167560,
    187366,
    181554,
    162401,
    187629,
    159887,
    187161,
    183850,
    171752,
    163826,
    171539,
    180381,
    181411,
    180372,
    172764,
    137327,
    11470,
    180220,
    187051,
    184357,
    164200,
    183110,
    183836,
    171785,
    177483,
    180209,
    181959,
    172836,
    169078,
    153580,
    189260,
    151151,
    182001,
    190395,
    140656,
    180169,

]
remain_codes = [155716, 136872, 163533, 174065, 179482, 167613]

if __name__ == '__main__':
    p = Pool(6)
    p.map(do_crawling, remain_codes)
