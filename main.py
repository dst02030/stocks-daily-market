import logging
import os
import ta
import numpy as np
import pandas as pd

from datetime import timedelta
from logging.handlers import TimedRotatingFileHandler

from src.utils import get_jinja_yaml_conf, create_db_engine, Clickhouse_client, Postgres_connect
from tqdm.auto import tqdm

def main():
    os.chdir(os.path.dirname(__file__))
    conf = get_jinja_yaml_conf('./conf/logging.yml', './conf/data.yml')
    tqdm.pandas()


    # logger 설정
    stream = logging.StreamHandler()
    # stream.setLevel(logging.DEBUG)
    logger = logging.getLogger('main')
    logging.basicConfig(level=eval(conf['logging']['level']),
        format=conf['logging']['format'],
        handlers = [TimedRotatingFileHandler(filename =  conf['logging']['file_name'],
                                    when=conf['logging']['when'],
                                    interval=conf['logging']['interval'],
                                    backupCount=conf['logging']['backupCount']), 
                                    stream]
                    )

    # DB 설정
    engine = create_db_engine(os.environ)
    postgres_conn = Postgres_connect(engine)
    click_conn = Clickhouse_client(user_name = os.environ['CLICK_USER'], password = os.environ['CLICK_PW'])
    
    # full_save 모드 확인 및 DB 최신 날짜 가져오기
    full_save = True if click_conn.get_count('stocks', 'daily_market') == 0 else os.environ['full_save'].lower() == 'true'

    logger.info(f'save mode is: {full_save}')

    # 마켓 정보 가져오기
    market_info = postgres_conn.get_data(conf['idx_market']['database'], conf['idx_market']['table'], 
                                columns = '*',
                                  orderby_cols = ['기준일자', '계열구분', '지수명']).rename(columns = {'대비': '전일대비', '등락률': '수익률', '상장시가총액': '시가총액'})
    

    market_dates = market_info['기준일자'].sort_values().unique()

    # 처리할 날짜 설정
    latest_market_date = pd.to_datetime(
                            click_conn.get_maxmin_col(conf['daily_market']['database'], conf['daily_market']['table'], 
                                column = '기준일자', is_min = False)[0]
        ).date()
    
    
    upload_date = market_dates[0] if full_save else latest_market_date + timedelta(days = 1)
    
    if upload_date > market_dates[-1]:
        logger.info("Latest market information is uploaded already.")
        return 
    
    start_idx = np.where(market_dates >= upload_date)[0][0] - max(conf['agg_days'])
    start_date = market_dates[0] if start_idx < 0 else market_dates[start_idx]
    
    logger.info(f"Upload from date: {upload_date}. For preprocessing, load date from {start_date}.")
    
    

    # 이평선 관련
    logger.info("Calculates market additional indicator.")
    group_info = market_info.groupby(['계열구분', '지수명'])
    for day in conf['agg_days']:
        logger.info(f"aggregating process of {day}days starts!")
        ## 종가
        # 이평
        market_info[f'종가_이평{day}일'] = group_info['종가'].rolling(window=day).mean().reset_index(level = [0, 1]).iloc[:, -1]
        # 괴리율
        market_info[f'종가_이평{day}일_괴리율'] = market_info['종가'] / market_info[f'종가_이평{day}일'] * 100
    
        ## 거래량
        # 이평
        market_info[f'거래량_이평{day}일'] = group_info['거래량'].rolling(window=day).mean().reset_index(level = [0, 1]).iloc[:, -1]
        # 합
        market_info[f'거래량_합{day}일'] = group_info['거래량'].rolling(window=day).sum().reset_index(level = [0, 1]).iloc[:, -1]
        
        ## 시총이평
        market_info[f'시총_이평{day}일'] = group_info['시가총액'].rolling(window=day).mean().reset_index(level = [0, 1]).iloc[:, -1]
        # 거래대금이평
        market_info[f'거래대금_이평{day}일'] = group_info['거래대금'].rolling(window=day).mean().reset_index(level = [0, 1]).iloc[:, -1]
        # 거래대금 합
        market_info[f'거래대금_합{day}일'] = group_info['거래대금'].rolling(window=day).sum().reset_index(level = [0, 1]).iloc[:, -1]
        
        ## 수익률
        # 이평
        market_info[f"수익률{day}일"] = group_info['수익률'].rolling(day).progress_apply(lambda x: ((1+x/100).prod() - 1) * 100, raw = True).reset_index(level = [0, 1]).iloc[:, -1]
        # 변동성
        market_info[f'수익률_변동성{day}일'] = group_info['수익률'].rolling(window=day).std().reset_index(level = [0, 1]).iloc[:, -1]
        # 이평 연율화
        market_info[f'수익률{day}일_연율화'] = ((1 + market_info[f"수익률{day}일"] / 100) ** (240 / 720) - 1) * 100
        # 변동성 연율화
        market_info[f'수익률_변동성{day}일_연율화'] = market_info[f'수익률_변동성{day}일'] / np.sqrt(240)
        # sr 연율화
        market_info[f'SR_{day}일_연율화'] =  market_info[f'수익률{day}일_연율화'] / market_info[f'수익률_변동성{day}일_연율화']
    
    # n일 최고/최저가
    for day in conf['high_low_days']:
        market_info[f'최고가{day}일'] = group_info['고가'].rolling(window=day).max().reset_index(level = [0, 1]).iloc[:, -1]
        market_info[f'최저가{day}일'] = group_info['저가'].rolling(window=day).min().reset_index(level = [0, 1]).iloc[:, -1]
        # 괴리율
        market_info[f'최고가{day}일_괴리율'] = market_info['종가'] / market_info[f'최고가{day}일'] * 100
        market_info[f'최저가{day}일_괴리율'] = market_info['종가'] / market_info[f'최저가{day}일'] * 100
    
    
    # 거래량
    market_info[f'거래량1일_증가율'] = market_info.apply(lambda x: np.nan if x['거래량_이평20일'] == 0 else x['거래량'] / x['거래량_이평20일'], axis = 1)
    market_info[f'거래량5일_증가율'] = market_info.apply(lambda x: np.nan if x['거래량_이평20일'] == 0 else x['거래량_이평5일'] / x['거래량_이평20일'], axis = 1)
    
    # 거래대금시총비율
    market_info[f'거래대금_시총비율_1일'] = market_info[f'거래대금'] / market_info[f'시가총액'] * 100
    market_info[f'거래대금_시총비율_5일'] = market_info[f'거래대금_이평5일'] / market_info[f'시총_이평5일'] * 100


    # 거래대금증가율
    market_info[f'거래대금_전일대비_증가율'] = market_info[f'거래대금'] / group_info['거래대금'].shift(1) * 100
    market_info[f'거래대금_5일이평대비_증가율'] = market_info[f'거래대금'] / market_info[f'거래대금_이평5일'] * 100
    
    
    # 기술적 지표 생성
    market_info['MACD'] = group_info.progress_apply(lambda x: ta.trend.macd(close = x['종가'], window_slow = 26, window_fast = 12)).reset_index(level = [0, 1]).iloc[:, -1]
    market_info['MACD_signal'] = group_info.progress_apply(lambda x: ta.trend.macd_signal(close = x['종가'], window_slow = 26, window_fast = 12, window_sign = 9)).reset_index(level = [0, 1]).iloc[:, -1]
    market_info['MACD_diff_signal'] = market_info['MACD'] - market_info['MACD_signal']
    market_info['RSI'] = group_info.progress_apply(lambda x: ta.momentum.rsi(close = x['종가'], window = 14)).reset_index(level = [0, 1]).iloc[:, -1]
    
    # Stochastic
    market_info['fastK'] = group_info.progress_apply(lambda x: ta.momentum.stoch(high = x['고가'], low = x['저가'], close = x['종가'], window=14, smooth_window=1)).reset_index(drop = True)
    market_info['fastD'] = market_info.groupby(['계열구분', '지수명'])['fastK'].rolling(window=3).mean().reset_index(level = [0, 1]).iloc[:, -1]
    market_info['slowK'] = market_info['fastD'].copy()
    market_info['slowD'] = market_info.groupby(['계열구분', '지수명'])['slowK'].rolling(window=3).mean().reset_index(level = [0, 1]).iloc[:, -1]
    
    
    # 볼린저밴드
    market_info['mavg'] = group_info.progress_apply(lambda x: ta.volatility.bollinger_mavg(close = x['종가'], window = 20)).reset_index(level = [0, 1]).iloc[:, -1]
    market_info['up'] = group_info.progress_apply(lambda x: ta.volatility.bollinger_hband(close = x['종가'], window = 20, window_dev = 2)).reset_index(level = [0, 1]).iloc[:, -1]
    market_info['dn'] = group_info.progress_apply(lambda x: ta.volatility.bollinger_lband(close = x['종가'], window = 20, window_dev = 2)).reset_index(level = [0, 1]).iloc[:, -1]
    
    # 골든 데드
    for cross_name, cross_cols in conf['tech_signal'].items():
        market_info[cross_name] = 0
        left = market_info.groupby(['계열구분', '지수명'])[cross_cols[0]]
        right = market_info.groupby(['계열구분', '지수명'])[cross_cols[1]]
        market_info.loc[(left.shift(1) <= right.shift(1)) & (left.shift(0) > right.shift(0)), cross_name] = 1 
        market_info.loc[(left.shift(1) >= right.shift(1)) & (left.shift(0) < right.shift(0)), cross_name] = -1
        market_info.loc[market_info[cross_cols[0]].isnull() | market_info[cross_cols[1]].isnull(), cross_name] = np.nan
    
    market_info['Bband_Cross'] = 0
    left = market_info.groupby(['계열구분', '지수명'])['종가']
    right = market_info.groupby(['계열구분', '지수명'])['up']
    market_info.loc[(left.shift(1) <= right.shift(1)) & (left.shift(0) > right.shift(0)) & (market_info['up'] > market_info['mavg'] * 1.15), 'Bband_Cross'] = 1
    right = market_info.groupby(['계열구분', '지수명'])['dn']
    market_info.loc[(left.shift(1) >= right.shift(1)) & (left.shift(0) < right.shift(0)) & (market_info['dn'] > market_info['mavg'] * 1.15), 'Bband_Cross'] = -1
    market_info.loc[market_info['up'].isnull() | market_info['dn'].isnull() | market_info['mavg'].isnull(), 'Bband_Cross'] = np.nan
    
    # 그외 전처리
    market_info['_ts'] = os.environ['_ts']
    market_info = market_info[market_info['기준일자'] >= upload_date].copy()


    # 데이터 업로드
    click_conn.df_insert(market_info, conf['daily_market']['database'], conf['daily_market']['table'])
        
if __name__ == "__main__":
    main()