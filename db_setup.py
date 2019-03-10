import numpy as np
import pandas as pd

import psycopg2
from configparser import ConfigParser

from sqlalchemy import create_engine

COURSE_TAGS = 'data_files/course_tags.csv'
USER_ASSESSMENT_SCORES = 'data_files/user_assessment_scores.csv'
USER_INTERESTS = 'data_files/user_interests.csv'
USER_COURSE_VIEWS = 'data_files/user_course_views.csv'

def config(filename='database.ini', section='postgresql'):
    '''
    >>> config()
    {'host': 'localhost', 'database': 'pluralsight', 'user': 'postgres', 'password': 'postgres', 'port': '5432'}
    '''
    parser = ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db

def loadCSV(file, **kwargs):
    '''
    >>> loadCSV('data_files/test.csv')
    Empty DataFrame
    Columns: [test]
    Index: []
    '''
    df = pd.read_csv(file, **kwargs)
    return df

def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE user_assessment_scores (
            user_handle INTEGER NOT NULL,
            assessment_tag VARCHAR(255) NOT NULL,
            user_assessment_date TIMESTAMP NOT NULL,
            user_assessment_score INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE user_interests (
            user_handle INTEGER NOT NULL,
            interest_tag VARCHAR(255) NOT NULL,
            date_followed TIMESTAMP NOT NULL
        )
        """,
        """
        CREATE TABLE course_tags (
            course_id VARCHAR(255) NOT NULL,
            course_tags VARCHAR(255) NOT NULL
        )
        """,
        """
        CREATE TABLE user_course_views (
            user_handle INTEGER NOT NULL,
            view_date DATE NOT NULL,
            course_id VARCHAR(255) NOT NULL,
            author_handle INTEGER NOT NULL,
            level VARCHAR(255) NOT NULL,
            view_time_seconds INTEGER NOT NULL
        )
        """)
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def upload_csv():
    """ copy csv to tables in the PostgreSQL database"""
    conn = None
    try:
        # params = config()
        # conn = psycopg2.connect(**params)
        # cur = conn.cursor()
        # for command in commands:
        #     cur.execute(command)
        # cur.close()
        # conn.commit()
        conn = create_engine('postgresql://postgres:postgres@localhost/pluralsight')

        dfUAS = loadCSV(USER_ASSESSMENT_SCORES)
        dfUI = loadCSV(USER_INTERESTS)
        dfCT = loadCSV(COURSE_TAGS)
        dfUCV = loadCSV(USER_COURSE_VIEWS)

        dfUAS.to_sql('user_assessment_scores', conn, if_exists='replace')
        dfUI.to_sql('user_interests', conn, if_exists='replace')
        dfCT.to_sql('course_tags', conn, if_exists='replace')
        dfUCV.to_sql('user_course_views', conn, if_exists='replace')

    except Exception as error:
        print(error)

def main():
    '''
    >>> main()
    ('PostgreSQL 11.2, compiled by Visual C++ build 1914, 64-bit',)
    '''
    params = config()
    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    cur.execute('SELECT version()')
    db_version = cur.fetchone()
    print(db_version)
    cur.close()

    # create_tables()
    upload_csv()


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    main()
