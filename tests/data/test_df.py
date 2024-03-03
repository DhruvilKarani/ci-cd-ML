import pytest
import pandas as pd


def test_df():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    assert 1 == 1