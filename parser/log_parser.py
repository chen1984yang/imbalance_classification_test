import pandas as pd

def parse_log(path):
    data = pd.read_csv(path, sep=",")
