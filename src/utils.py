import clickhouse_connect
import logging
import yaml

import re
import requests
import json

import numpy as np
import pandas as pd


from datetime import date, datetime
from sqlalchemy import create_engine
from jinja2 import Template
from urllib.parse import quote


logger = logging.getLogger(__name__)

def get_jinja_yaml_conf(*file_names):
    """
    jinja2 포맷을 이용해서 yaml 파일을 읽는 함수입니다.
    
    Args:
        file_name: 읽을 파일 이름입니다.
    
    Returns: 
        conf_dict: 파일을 읽어낸 결과입니다.
    """

    conf_dict = dict()

    for file_name in file_names:

        with open(file_name, encoding='utf-8') as f:
            t = Template(f.read())
            c = yaml.safe_load(t.render())
            def_conf = yaml.safe_load(t.render(c))
            conf_dict.update(def_conf)
            
    return conf_dict




def create_db_engine(engine_info):
    """
    DB 엔진을 만드는 함수입니다.

    Args:
        engine_info: DB 정보를 포함한 딕셔너리 입니다. DB_TYPE, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME 정보를 포함해야 합니다.

    Returns:
        engine: 생성된 DB 엔진입니다.
    """
    engine_info = {key: quote(val) for key, val in engine_info.items()}
    conn_str = "{DB_TYPE}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}".format(**engine_info)
    engine = create_engine(conn_str, client_encoding='utf8')
    logger.info(f'Connect to {engine_info["DB_HOST"]}. DB_NAME is {engine_info["DB_NAME"]}')
    return engine


class Postgres_connect:

    def __init__(self, engine):
        self.engine = engine
        return

    def get_data(self, schema_name, table_name, columns = [], where = [], where_operator = 'AND', additional_sql = "", is_distinct = False, orderby_cols = []):
        columns = [columns] if isinstance(columns, str) else columns
        
        if len(columns) == 0:
            columns = ['*']
        
        where_clause = self._make_where(where, where_operator)
        sql = f"""SELECT {"DISTINCT" if is_distinct else ""} {', '.join(columns)} 
                                FROM {schema_name}.{table_name} {where_clause}""" + additional_sql
        
        data = pd.read_sql(sql, con = self.engine
                                )

        logger.debug(sql)

        if orderby_cols:
            orderby_cols = [orderby_cols] if isinstance(orderby_cols, str) else orderby_cols
            data.sort_values(by = orderby_cols, ignore_index = True, inplace = True)


        return data


    

    def get_maxmin_col(self, schema_name, table_name, column, where = [], where_operator = 'AND', additional_sql = "", is_max = True, is_min = True):
        if not is_max and not is_min:
            raise Exception("You should set True at least one of is_min or is_max.")

        where_clause = self._make_where(where, where_operator)
        maxmin = []
        if is_max: maxmin.append(f"MAX({column})")
        if is_min: maxmin.append(f"MIN({column})")

        sql = f"""SELECT {", ".join(maxmin)} 
                                FROM {schema_name}.{table_name} {where_clause}""" + additional_sql
        logger.debug(sql)

        return pd.read_sql(sql, con = self.engine
                                ).to_numpy().ravel()

    

    def get_count(self, schema_name, table_name, where = [], where_operator = 'AND', additional_sql = ""):
        where_clause = self._make_where(where, where_operator)
        sql = f"SELECT COUNT(*) FROM {schema_name}.{table_name} {where_clause}" + additional_sql
        data = pd.read_sql(sql, con = self.engine
                                ).iloc[0].item()
        logger.debug(sql)
        return int(data)
    
    
    
    def insert_df(self, data, schema_name, table_name):
        data.to_sql(con = self.engine,
                schema = schema_name,
                name = table_name,
                if_exists = 'append',
                index = False)
        
        logger.info(f"Upload data successfully (rows: {data.shape[0]}).")
        
        
        
    
    def upsert(self, data, schema_name, table_name, del_rows = 1000):

        # 해당 테이블의 pkey constraint 이름 찾기
        pkey_constraint = pd.read_sql(sql = f"""SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE table_schema = '{schema_name}' AND table_name = '{table_name}' AND constraint_type = 'PRIMARY KEY'""",
            con = self.engine)
        
        if pkey_constraint.shape[0] == 0:
            logger.info(f"Upload data starts: {data.shape[0]} rows.")
            return data.to_sql(table_name, schema = schema_name, con = self.engine, index = False, if_exists = "append")

        else:
            pkey_constraint = pkey_constraint.iloc[0, 0]

        # 해당 테이블의 primary key 리스트
        pkey_cols = pd.read_sql(f"""SELECT column_name
        FROM information_schema.key_column_usage
        WHERE table_schema = '{schema_name}' AND table_name = '{table_name}' AND constraint_name = '{pkey_constraint}';""",
                con = self.engine).to_numpy().ravel()
        
        # db 내에 pkey 중복 데이터 제거를 위한 where문 생성
        in_list = list()
        for _, row in data[pkey_cols].iterrows():
            row_processing = [val.replace("'", "''").replace('%', '%%') if isinstance(val, str) else val for val in row.values]
            row_processing = [f"'{val}'" if isinstance(val, (str, date, datetime)) else str(val) for val in row_processing]
            in_list.append(f"({', '.join(row_processing)})")
        
        
        # delete문을 통해 중복 데이터 제거
        for i in range(len(in_list) // del_rows + 1):
            with self.engine.begin() as conn:
                del_list = in_list[i*del_rows:(i+1)*del_rows]
                where_clause = f"WHERE ({', '.join(pkey_cols)}) IN ({', '.join(del_list)})"

                del_sql = f"""DELETE FROM {schema_name}.{table_name}
                                            {where_clause}"""

                if len(del_list) > 0:
                    logger.debug(del_sql)
                    conn.exec_driver_sql(del_sql)

        logger.info(f"Upload data starts: {data.shape[0]} rows.")
        
        # 데이터 업로드
        return data.to_sql(table_name, schema = schema_name, con = self.engine, index = False, if_exists = "append")
    

    
    def ext_notin_db(self, data, schema_name, table_name,  subset = []):

        if isinstance(subset, str):
            subset = [subset]
        
        if len(subset) == 0:
            subset = data.columns.tolist()


        db_data = self.get_data(schema_name = schema_name, 
                        table_name = table_name, 
                        columns = subset,
                        is_distinct = True)
        
        coltype = db_data.dtypes
        coltype[coltype =='object'] = str
        data = data.astype(coltype)

        merge_data = pd.merge(data, db_data, how = 'outer', on = subset, indicator = True)
        

        return merge_data[merge_data['_merge'] == 'left_only'].drop(columns = '_merge')
    

    def get_engine(self):
        return self.engine

    def _make_where(self, where, where_operator = 'AND'):
        where = [where] if isinstance(where, str) and where != '' else [w for w in where if w != '']
        return f"WHERE {f' {where_operator} '.join(where)}" if where else ""


class Clickhouse_client:
    """
    Clickhouse Database에 연결해 client를 생성하는 class입니다.
    기존 clickhouse driver Client 인스턴스의 함수 한계를 개선하였습니다.
    """

    def __init__(self, host = '172.20.10.2', settings = {}, user_name ='default', password = '', *args, **kwargs):
        self._client = clickhouse_connect.get_client(host=host, settings = settings, user_name = user_name, password = password)
        self.type_converter = {'DateTime': 'datetime64[ns]', 'Date': 'datetime64[ns]', 'String': str, 'Float': float, 'Int': 'Int64', 'Decimal': float, 'UInt': 'Int64', 'Nothing': float}

    def __del__(self):
        self._client.close()

    def table_description(self, schema_name, table_name, *args, **kwargs):
        """
        db 테이블 구조를 출력하는 함수입니다.
        Args:
            schema_name: 입력할 db schema명
            table_name: 입력할 db table명
        Returns:
            table schema 구조 출력
        """
        return self.execute(f'DESCRIBE TABLE {schema_name}.{table_name}')


    def match_python_type(self, schema_name, table_name, *args, **kwargs):
        table_info = self.table_description(schema_name, table_name)
        column_type = {col[1]['name']: self.type_converter[re.sub(r'Nullable\(|\([^)]*\)|[0-9\, \(\)]|Fixed', '', col[1]['type'])] for col in table_info.iterrows()}
        return column_type


    def get_system_table(self, schema_name = '', table_name = '', columns = [], *args, **kwargs):
        """
        db에 저장된 system table을 출력하는 함수입니다.
        Args:
            schema_name: 입력할 db schema명
            table_name: 입력할 db table명
        Returns:
            (list) table sorting key를 출력
        """
        
        if table_name == '':
            raise ValueError("table명을 입력하세요")

        where_clause = f"WHERE name = '{table_name}'"

        if isinstance(schema_name, str) and len(schema_name)>0:
            where_clause += f" AND database = '{schema_name}'"

        if isinstance(columns, str):
            columns = [columns]

        if len(columns) == 0:
            columns = ["*"]

        wrapped_columns = self._column_wrapping(columns)

        return self.execute(f"SELECT {', '.join(wrapped_columns)} FROM system.tables {where_clause}")

    def df_processing(self, df, schema_name, table_name, *args, **kwargs):
        """
        db table을 입력하거나 출력할 때 값을 처리하는 함수입니다.
        Args:
            df: 입력할 데이터프레임

        Returns:
            df: 처리된 데이터프레임
        """
        logger.info(f"data processing is started!")
        column_type = self.match_python_type(schema_name, table_name)
        column_type = {key: val for key, val in column_type.items() if key in df.columns}
        df = df.loc[:, column_type.keys()].copy()
        
        # Datetime 형들은 timezone을 제거
        datetime_col = [key for key, val in column_type.items() if val == 'datetime64[ns]']
        df[datetime_col] = df[datetime_col].map(lambda x: pd.to_datetime(x)).map(lambda x: x.tz_convert('UTC').tz_localize(None) if x.tzname() else x)


        # int 형 보장 및 형변환 후 테이블 칼럼 순서대로 정렬
        # nan 값을 None으로 치환(파이썬의 None type이 clickhouse에서는 null타입으로 입력됨)
        int_col = [key for key, val in column_type.items() if val == "Int64"]
        df[int_col] = df[int_col].astype(float) // 1
        df = df.astype(column_type).replace([np.inf, -np.inf, 'NaT', 'NaN', 'nan', '<NA>', 'null', 'N/A', "None"], np.nan).where(lambda x: x.notnull(), None)

        # sorting key인 칼럼들은 na값을 대체
        sorting_key = self.get_table_key(schema_name = schema_name, table_name = table_name)
        df.fillna({col: "" for col in sorting_key if col in df.columns and df[col].dtypes == 'object'}, inplace = True)


        logger.info(f"data processing is finished.")

        return df

    def df_insert(self, df, schema_name, table_name, chunk_size = 1000000, settings = {}, *args, **kwargs):
        """
        db table에 데이터프레임을 insert하는 함수입니다.
        Args:
            df: 입력할 데이터프레임
            schema_name: 입력할 db schema명
            table_name: 입력할 db table명
            chunk_size: 입력 단위

        Returns:
            count: db에 입력한 데이터 row 수
        """

        if df.shape[0] == 0:
            logger.warning("There is no data in dataframe. Skip insert process")
            return

        logger.info(f'df insert to db starts!, schema: {schema_name}, table: {table_name}.')



        df = self.df_processing(df, schema_name, table_name)

        chunk_num, count = df.shape[0] // chunk_size + 1, 0

        for i in range(chunk_num):
            logger.debug(self._client.insert_df(df = df.iloc[i*chunk_size:(i+1)*chunk_size], database = schema_name, table = table_name, settings = settings).summary)
            count += df.iloc[i*chunk_size:(i+1)*chunk_size].shape[0]
            logger.info(f"data insert is processing ({count}/{df.shape[0]}).")

        logger.info(f"data insert is finished.")

        return count



    def get_table(self, schema_name, table_name, columns = ["*"], where = [], orderby_cols = [], post_sql = '', settings = {}, *args, **kwargs):

        if columns == '*':
            columns = self.table_description(schema_name, table_name)['name'].tolist()
        wrapped_columns = self._column_wrapping(columns)
        where = [where] if isinstance(where, str) and where != '' else [w for w in where if w != '']

        orderby_cols = [orderby_cols] if isinstance(orderby_cols, str) and orderby_cols != '' else [o for o in orderby_cols if o != '']
        orderby_cols = self._column_wrapping(orderby_cols)
        where_clause = 'WHERE ' + ' AND '.join(where) if len(where) > 0 else ''
        orderby_clause = 'ORDER BY ' + ', '.join(orderby_cols) if len(orderby_cols) > 0 else ''

        sql = f"""SELECT {', '.join(wrapped_columns)} FROM {schema_name}.{table_name} FINAL """ + where_clause + orderby_clause + post_sql
        data = self.execute(sql, settings = settings)


        if data.shape[0] == 0:
            logger.warning("There are no data that fulfill condition.")
            return pd.DataFrame(columns = columns)

        data = self.df_processing(data, schema_name, table_name)

        return data

    def get_maxmin_col(self, schema_name, table_name, column, where = [], where_operator = 'AND', additional_sql = "", is_max = True, is_min = True):
        if not is_max and not is_min:
            raise Exception("You should set True at least one of is_min or is_max.")

        where_clause = self._make_where(where, where_operator)
        maxmin = []
        if is_max: maxmin.append(f"MAX(`{column}`)")
        if is_min: maxmin.append(f"MIN(`{column}`)")

        sql = f"""SELECT {", ".join(maxmin)} 
                                FROM {schema_name}.{table_name} {where_clause}""" + additional_sql
        
        data = self.execute(sql).to_numpy().ravel()

        return data


    def get_client(self, *args, **kwargs):
        """
        생성한 client를 출력
        """
        return self._client

    def execute(self, sql, settings = {}, *args, **kwargs):
        """
        입력한 sql을 실행하는 함수
        """
        logger.info(f'sql execute: {sql}')
        return self._client.query_df(sql, query_formats={'FixedString': 'string'}, settings = settings)

    def get_count(self, schema_name, table_name, columns = ["*"], where = [], orderby_cols = [], post_sql = ''):
        """
        테이블 내 특정 조건에 해당하는 row 수를 출력하는 함수
        """
        if columns == '*':
            columns = self.table_description(schema_name, table_name)['name'].tolist()
        columns = self._column_wrapping(columns)
        where = [where] if isinstance(where, str) and where != '' else [w for w in where if w != '']

        orderby_cols = [orderby_cols] if isinstance(orderby_cols, str) and orderby_cols != '' else [o for o in orderby_cols if o != '']
        orderby_cols = self._column_wrapping(orderby_cols)
        where_clause = 'WHERE ' + ' AND '.join(where) if len(where) > 0 else ''
        orderby_clause = 'ORDER BY ' + ', '.join(orderby_cols) if len(orderby_cols) > 0 else ''
        
        return self.execute(f"SELECT COUNT({', '.join(columns)}) FROM {schema_name}.{table_name} FINAL" + where_clause + orderby_clause + post_sql).iloc[0, 0]




    def get_distinct_rows(self, schema_name, table_name, dist_cols = '*', where = [], orderby_cols = [], post_sql = ''):
        dist_cols = [dist_cols] if isinstance(dist_cols, str) else dist_cols
        dist_cols = self._column_wrapping(dist_cols)



        orderby_cols = [orderby_cols] if isinstance(orderby_cols, str) and orderby_cols != '' else [o for o in orderby_cols if o != '']
        orderby_cols = self._column_wrapping(orderby_cols)
        where_clause = 'WHERE ' + ' AND '.join(where) if len(where) > 0 else ''
        orderby_clause = 'ORDER BY ' + ', '.join(orderby_cols) if len(orderby_cols) > 0 else '' 
        
        data = self.execute(f"""SELECT DISTINCT {', '.join(dist_cols)} FROM {schema_name}.{table_name} FINAL
                                {where_clause} {orderby_clause} {post_sql}""")
        return data


    def get_table_key(self, schema_name, table_name):
        info = self.get_system_table(schema_name = schema_name, table_name = table_name, columns = ['sorting_key'])
        if info.shape[0] == 0:
            raise ValueError(f"{schema_name}.{table_name} cannot be found in database. Please check database and table name.")

        elif info.shape[0] > 1:
            raise IndexError(f"There is more than one table with the name {table_name}. Please specify the database.")

        return info.iloc[0, 0].split(', ')


    def make_single_quote_escape(self, vals):
        """
        입력된 vals에 존재하는 single quote(')를 sql문에서 입력가능하게 변경하는 함수. escape(\\)를 이용함.
        """
        if isinstance(vals, str):
            vals = [vals]

    def _column_wrapping(self, cols):
        new_cols = []
        for col in cols:
            if re.search('[가-힣]', col):
                col = col.split()
                col = f"`{col[0]}`" if len(col) <= 1 else f"`{col[0]}` {' '.join(col[1:])}"
                new_cols.append(col)

            elif len(col) > 0:
                new_cols.append(col)
        
        return new_cols

    def _make_where(self, where, where_operator = 'AND'):
        where = [where] if isinstance(where, str) and where != '' else [w for w in where if w != '']
        return f"WHERE {f' {where_operator} '.join(where)}" if where else ""