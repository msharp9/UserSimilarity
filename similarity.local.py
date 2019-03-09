import os
import datetime

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

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

    dfLevel = pd.crosstab(dfUCV.user_handle, dfUCV.level)
    dfAuthors = pd.crosstab(dfUCV.user_handle, dfUCV.author_handle)

    dfAssTags = pd.crosstab(dfUAS.user_handle, dfUAS['assessment_tag'])
    dfUserTags = pd.crosstab(dfUI.user_handle, dfUI.interest_tag).gt(0).astype(int)

    dfUCV2 = dfUCV.drop(columns=['view_time_seconds', 'view_date']).drop_duplicates()
    dfUCV2 = dfUCV2.merge(dfCT, on=['course_id'], how='left')
    dfViewTags = pd.crosstab(dfUCV2.user_handle, dfUCV2.course_tags)

    dfUsers = pd.concat([dfAssScores, dfUserViews, dfLevel, dfAuthors, dfAssTags, dfUserTags, dfViewTags], axis=1).fillna(0)
    return dfUsers

def main():
    '''
    >>> main()
    starting...
    '''
    print('starting...')
    dfUsers = userFeatures()


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    main()
