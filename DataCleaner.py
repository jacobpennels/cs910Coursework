import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class DataCleaner:
    def draw_histogram(self, d):
        labels, values = zip(*Counter(d).items())
        indexs = np.arange(len(labels))
        width = 1

        plt.bar(indexs, values, width)
        plt.xticks(indexs + width * 0.5, labels)
        plt.show()

    def correct_ages(self, d):
        d['date_of_birth'] = pd.to_datetime(d['date_of_birth'], format='%Y-%m-%d')
        d['datetime'] = pd.to_datetime(d['datetime'], format='%Y-%m-%d')
        d['days_at_outcome'] = (d['datetime'] - d['date_of_birth']).dt.days
        return d

    def fill_missing_values(self, d, column, new_val):
        d[column] = d[column].fillna(new_val)
        return d

    def clean_data(self, d):
        d = self.correct_ages(d)
        d = self.fill_missing_values(d, 'name', 'no-name')
        d = self.fill_missing_values(d, 'outcome_subtype', 'none')
        d = d.drop('animal_id', axis=1)
        d = d.replace(regex=r'\*+', value="")
        return d

if __name__ == '__main__':
    data = pd.read_csv('../datafiles/aac_shelter_outcomes.csv')

    #draw_histogram(data['age_upon_outcome'])

    data = correct_ages(data)

    data = fill_missing_values(data, 'name', 'no-name')
    data = fill_missing_values(data, 'outcome_subtype', 'none')
    data = data.drop('animal_id', axis=1)
    data = data.replace(regex=r'\*+', value="")



