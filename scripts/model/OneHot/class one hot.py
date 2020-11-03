import os
import sys
import pandas as pd
import scipy
from pymongo import MongoClient
from scripts.mongoConnection import

class one_hot:
    """One Hot class provides function to apply OneHot encoding using logistic regression on the given training and testing data to
    generate prediction with accuracy functions
        """
    def __init__(self, filepath, score_function):
        self.filepath = filepath
        self.score_function = score_function
        self.my_tags = ['belonging', 'meaning', 'efficacy', 'distinctivness', 'self esteem', 'continuity']


