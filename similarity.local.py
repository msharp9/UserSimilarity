import os
# import datetime

import numpy as np
import pandas as pd

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

COURSE_TAGS = 'data_files/course_tags.csv'
USER_ASSESSMENT_SCORES = 'data_files/user_assessment_scores.csv'
USER_INTERESTS = 'data_files/user_interests.csv'
USER_COURSE_VIEWS = 'data_files/user_course_views.csv'

USERID = 1

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
    [4065, 1872, 9532, 4411, 1526, 9966, 7173, 7663, 5991, 802]
    '''
    dfUsers = userFeatures()
    # dfUsers.to_csv('data_files/similarity.csv')
    sim = Similarity(dfUsers)
    return sim.likeUsers(USERID)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    main()
