import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def normalize_data(X):
    # Normalização dos dados (entre 0 e 1)
    return (X - X.min()) / (X.max() - X.min())

def train_test_split(X, y, test_size=0.3):
    # Dividindo os dados em treino e teste
    test_size = int(len(X) * test_size)
    X_train = X[:len(X) - test_size]
    X_test = X[len(X) - test_size:]
    y_train = y[:len(y) - test_size]
    y_test = y[len(y) - test_size:]
    return X_train, X_test, y_train, y_test
