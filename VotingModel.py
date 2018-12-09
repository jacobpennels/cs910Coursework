import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from project.DataCleaner import DataCleaner
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

class VotingModel:
    def __init__(self, d):
        self.data = d
        self.cols = ['animal_type', 'breed', 'color', 'name', 'outcome_subtype', 'sex_upon_outcome',
                     'days_at_outcome']
        self.class_col = 'outcome_type'

    def test_model(self, k, d_train, d_test):
        nb = GaussianNB()
        dc = DecisionTreeClassifier()
        km = KNeighborsClassifier(k)

        vc = VotingClassifier(estimators=[('nb', nb), ('dc', dc), ('km', km)], voting='soft')
        vc.fit(d_train[self.cols], d_train[self.class_col])
        return vc.score(d_test[self.cols], d_test[self.class_col])

    def do_kfold(self, kmin, kmax, folds):
        if (len(self.data) < folds):
            folds = len(self.data)
        kf = KFold(n_splits=folds)
        best_avg = 0
        best_k = kmin
        for i in range(kmin, kmax + 1):
            if i > len(self.data) - (len(self.data)) / folds:
                break
            print("Testing for k=" + str(i))
            accuracy = 0
            for train, test in kf.split(self.data):
                accuracy += self.test_model(i, self.data.iloc[train], self.data.iloc[test])
            accuracy /= folds
            if accuracy > best_avg:
                best_avg = accuracy
                best_k = i
            print("Accuracy: " + str(accuracy))
        return best_k, best_avg


if __name__ == '__main__':
    data = pd.read_csv('../datafiles/aac_shelter_outcomes.csv')
    dc = DataCleaner()
    data = dc.clean_data(data)
    categoric_attributes = ['animal_type', 'breed', 'color', 'name', 'outcome_subtype', 'outcome_type',
                            'sex_upon_outcome']
    data, encodings = dc.label_encode(data, categoric_attributes)

    nb = VotingModel(data)
    k, acc = nb.do_kfold(1, 30, 10)
    print("Best accuracy: " + str(acc) + ", where K-NN k=" + str(k))