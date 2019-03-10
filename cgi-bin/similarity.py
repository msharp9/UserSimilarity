# !/usr/bin/python
import os
import sys
# import datetime

import numpy as np
import pandas as pd

import psycopg2
from configparser import ConfigParser

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import cgi
import json

def printError(msg):
    '''
    >>> printError('error')
    Content-Type: text/plain
    Status: 400 Bad Request
    <BLANKLINE>
    error
    '''
    print('Content-Type: text/plain')
    print('Status: 400 Bad Request')
    print()
    print(msg)

class Form():
    def __init__(self):
        self.form = cgi.FieldStorage()

    def checkForm(self, item):
        '''
        >>> Form().checkForm('')
        SystemExit: Script Input Failure
        '''
        if item in self.form:
            return self.form.getlist(item)
        else:
            printError('Error! Failed to provide input: ' + item)
            sys.exit('Script Input Failure')

    def checkOptionalForm(self, item):
        '''
        >>> Form().checkOptionalForm('test')
        []
        '''
        if item in self.form:
            return self.form.getlist(item)
        else:
            return []

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


COURSE_TAGS = 'data_files/course_tags.csv'
USER_ASSESSMENT_SCORES = 'data_files/user_assessment_scores.csv'
USER_INTERESTS = 'data_files/user_interests.csv'
USER_COURSE_VIEWS = 'data_files/user_course_views.csv'

def loadCSV(file, **kwargs):
    '''
    >>> loadCSV('data_files/test.csv')
    Empty DataFrame
    Columns: [test]
    Index: []
    '''
    df = pd.read_csv(file, **kwargs)
    return df

def userFeatures():
    '''
    create user feature DataFrame
    >>> userFeatures().shape
    (10000, 3220)
    '''
    params = config()
    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    cur.execute('SELECT version()')
    db_version = cur.fetchone()
    print(db_version)
    # cur.close()

    dfUAS = loadCSV(USER_ASSESSMENT_SCORES)
    dfUI = loadCSV(USER_INTERESTS)
    dfCT = loadCSV(COURSE_TAGS)
    dfUCV = loadCSV(USER_COURSE_VIEWS)



    dfAssScores = dfUAS.groupby('user_handle').agg(
            { 'user_assessment_score' : ['mean', 'count'] }
        )
    dfUserViews = dfUCV.groupby('user_handle').agg(
            {
                'view_time_seconds' : ['mean', 'sum', 'count'],
            }
        )
    sc = StandardScaler()
    dfAssScores['user_assessment_score'] = sc.fit_transform(dfAssScores['user_assessment_score'])
    dfUserViews['view_time_seconds'] = sc.fit_transform(dfUserViews['view_time_seconds'])

    dfLevel = pd.crosstab(dfUCV.user_handle, dfUCV.level)
    dfAuthors = pd.crosstab(dfUCV.user_handle, dfUCV.author_handle)

    dfAssTags = pd.crosstab(dfUAS.user_handle, dfUAS['assessment_tag'])
    dfUserTags = pd.crosstab(dfUI.user_handle, dfUI.interest_tag
        ).gt(0).astype(int)

    dfUCV2 = dfUCV.drop(columns=['view_time_seconds', 'view_date']
        ).drop_duplicates()
    dfUCV2 = dfUCV2.merge(dfCT, on=['course_id'], how='left')
    dfViewTags = pd.crosstab(dfUCV2.user_handle, dfUCV2.course_tags)

    dfUsers = pd.concat([dfAssScores, dfUserViews, dfLevel, dfAuthors,
        dfAssTags, dfUserTags, dfViewTags], axis=1).fillna(0)
    return dfUsers

class Similarity():
    def __init__(self, data, n_neighbors=10, algorithm='ball_tree',
        metric='minkowski', **kwargs):
        # pipe = Pipeline([('scl', StandardScaler()),
        #                 ('pca', PCA(n_components=10)),
        #                 ('nbrs', NearestNeighbors(n_neighbors=10,
        #                     algorithm='ball_tree'))
        #                 ])
        self.data = data
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors,
            algorithm=algorithm, metric=metric, **kwargs)
        self.nbrs.fit(data)

    def likeUsers(self, id, n=10):
        '''
        >>> sim = Similarity(pd.DataFrame([1,2,3,4,5]), 2)
        >>> sim.likeUsers(1,2)
        [2, 0]
        '''
        index = self.nbrs.kneighbors(self.data.loc[[id]], n_neighbors=n+1,
            return_distance=False)
        similar_users = self.data.iloc[index[0]].index.tolist()
        return similar_users[1:]

def main():
    '''
    >>> main()
    Content-Type: application/json
    <BLANKLINE>
    [6531, 1712, 6221, 329, 4092, 358, 1968, 1084, 1983, 9258]
    '''
    form = Form()
    userid = form.checkForm('userid')[0]
    userid = int(userid)
    n = form.checkOptionalForm('n')
    if n:
        n = n[0]
    else:
        n = 10

    dfUsers = userFeatures()
    sim = Similarity(dfUsers, n)
    like_users= sim.likeUsers(userid, n)

    print("Content-Type: application/json")
    print()
    print(json.dumps(like_users))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    main()
