import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from project.DataCleaner import DataCleaner
from sklearn.model_selection import KFold
from sklearn import tree

class DecisionTreeModel:
    def __init__(self, d):
        self.data = d
        self.cols = ['animal_type', 'breed', 'color', 'name', 'outcome_subtype', 'sex_upon_outcome',
                     'days_at_outcome']
        self.class_col = 'outcome_type'

    def create_tree_file(self, name):
        dc = tree.DecisionTreeClassifier()
        dc.fit(self.data[self.cols], self.data[self.class_col])
        print("Saving the tree")
        dot_data = tree.export_graphviz(dc, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(name)

    def test_model(self, d_train, d_test, save=False):
        dc = tree.DecisionTreeClassifier()
        dc.fit(d_train[self.cols], d_train[self.class_col])

        return dc.score(d_test[self.cols], d_test[self.class_col])

    def test_on_whole_model(self, t):
        dc = tree.DecisionTreeClassifier()
        dc.fit(self.data[self.cols], self.data[self.class_col])

        test_data = self.data.loc[self.data['animal_type'] == t]
        return dc.score(test_data[self.cols], test_data[self.class_col])

    def do_kfold(self, kmin, kmax, folds, save=False):
        if (len(self.data) < folds):
            folds = len(self.data)
        kf = KFold(n_splits=folds)
        best_avg = 0
        best_k = kmin
        for i in range(kmin, kmax + 1):
            if (i > len(self.data) - (len(self.data)) / folds):
                break
            # print("Testing for k=" + str(i))
            accuracy = 0
            for train, test in kf.split(self.data):
                accuracy += self.test_model(self.data.iloc[train], self.data.iloc[test], save)
            accuracy /= folds
            if (accuracy > best_avg):
                best_avg = accuracy
                best_k = i
        return best_k, best_avg


if __name__ == '__main__':
    data = pd.read_csv('../datafiles/aac_shelter_outcomes.csv')
    dc = DataCleaner()
    data = dc.clean_data(data)
    categoric_attributes = ['animal_type', 'breed', 'color', 'name', 'outcome_subtype', 'outcome_type',
                            'sex_upon_outcome']
    data, encodings = dc.label_encode(data, categoric_attributes)

    dc = DecisionTreeModel(data)
    '''
    k, acc = dc.do_kfold(1, 1, 10, True)
    print("Accuracy is " + str(acc))

    dc.create_tree_file('tree')
    
    avg_acc = 0
    for d in data['animal_type'].unique():
        actual_label = encodings['animal_type'].inverse_transform([d])[0]
        print("Testing for animal type " + actual_label)
        dc = DecisionTreeModel(data.loc[data['animal_type'] == d])
        k, acc = dc.do_kfold(1, 1, 10)
        print("The highest accuracy was " + str(acc) + ", where k=" + str(k))
        avg_acc += acc

    print("The average accuracy of the individual types is: " + str(avg_acc/5))
    '''
    t = encodings['animal_type'].transform(['Livestock'])[0]
    print("Livestock accuracy: " + str(dc.test_on_whole_model(t)))