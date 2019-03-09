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

def main():
    '''
    >>> main()
    starting...
    '''
    print('starting...')

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    main()
