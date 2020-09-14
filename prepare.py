import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def iris_split(df):

    train_validate, test = train_test_split(df, test_size=.2,
                                        random_state=123,
                                        stratify=df.species)
    train, validate = train_test_split(train_validate, test_size=.3,
                                        random_state=123,
                                        stratify=train_validate.species)
    return train, validate, test

    