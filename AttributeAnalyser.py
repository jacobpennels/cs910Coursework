import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from project.DataCleaner import DataCleaner
from collections import Counter, OrderedDict


data = pd.read_csv('../datafiles/aac_shelter_outcomes.csv')
dc = DataCleaner()
data = dc.clean_data(data)


def age_analysis(d):
    #print(min(d['days_at_outcome']))
    #print(max(d['days_at_outcome']))
    print(Counter(d['outcome_type']))
    plot_data = []
    labels = []
    for t in d['outcome_type'].unique():
        plot_data.append(d.loc[d['outcome_type'] == t, ['days_at_outcome']].values)
        labels.append(t)

    info = plt.boxplot(plot_data, labels=labels, showfliers=False)
    plt.ylabel("Age (Days)")

    plt.show()


def outcome_subtype_analysis(d):
    for t in d['outcome_subtype'].unique():
        if(t != 'none'):
            outcomes = Counter(d.loc[d['outcome_subtype'] == t, 'outcome_type'])
            print(t + " produces outcomes " + str(outcomes))


def create_pie_subplot(a, s, l, r, c, t):
    a[r, c].pie(s, labels=l)
    a[r, c].set_title(t)


def type_analysis(d):
    '''sizes = []
    labels = []
    counts = Counter(d['animal_type'])
    print(counts)
    for k, v in counts.items():
        labels.append(k)
        sizes.append(v)

    wedges, texts = plt.pie(sizes)
    plt.legend(wedges, labels, loc="center left", title="Animal Type", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()'''
    fig, axes = plt.subplots(2, 3)
    types = d['animal_type'].unique()
    type_count = 0
    for i in range(2):
        if(i == 1):
            loops = 2
        else:
            loops = 3
        for j in range(loops):
            c_t = types[type_count]
            type_count += 1
            sizes = []
            labels = []
            counts = Counter(d.loc[d['animal_type'] == c_t, 'outcome_type'])
            for k, v in counts.items():
                sizes.append(v)
                labels.append(k)
            create_pie_subplot(axes, sizes, labels, i, j, c_t)
    axes[-1, -1].axis('off')
    plt.figlegend()
    plt.show()


def name_analysis(d):
    popular_names = [
        "Max", "Charlie", "Buddy", "Cooper", "Rocky", "Jack",  "Jake",  "Toby", "Bailey", "Oliver", "Bentley", "Tucker",
        "Duke", "Teddy", "Cody", "Riley", "Bear", "Buster", "Murphy", "Harley", "Bella", "Lucy", "Daisy", "Molly",
        "Maggie", "Lola", "Sophie", "Chloe", "Sadie", "Bailey", "Coco", "Lily", "Gracie", "Roxy", "Abby", "Zoey",
        "Stella", "Zoe", "Ginger", "Penny", "Oliver", "Max", "Charlie", "Jack", "Simba", "Leo", "Milo", "Tiger",
        "Smokey", "Buddy", "Tigger", "Sammy", "Toby", "Oscar", "Shadow", "Sam", "Simon", "Jasper", "Oreo", "Rocky",
        "Bella", "Chloe", "Lucy", "Lily", "Sophie", "Luna", "Gracie", "Molly", "Zoe", "Cleo", "Mia", "Princess",
        "Daisy", "Abby", "Sasha", "Callie", "Angel", "Kitty", "Lola", "Maggie"
    ]
    popular_names = set(popular_names)
    no_name_data = d.loc[d['name'] == "no-name", ['name',"outcome_type"]]

    popular_name_data = d[d['name'].isin(popular_names)][['name','outcome_type']]

    indexes = list(no_name_data.index.values) + list(popular_name_data.index.values)

    unpopular_name_data = d.drop(indexes)[['name', 'outcome_type']]
    counts = Counter(no_name_data['outcome_type'])
    od = OrderedDict(sorted(counts.items()))
    print(od)
    sizes = []
    labels = []
    for k, v in od.items():
        labels.append(k)
        sizes.append(v)

    wedges, texts = plt.pie(sizes)
    plt.legend(wedges, labels, title="Outcome", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title("Animals with no names")
    plt.show()


def sex_analysis(d):
    '''
    counts = Counter(d['sex_upon_outcome'])
    sizes = []
    labels = []
    for k, v in counts.items():
        labels.append(k)
        sizes.append(v)
    wedges, texts = plt.pie(sizes)
    plt.legend(wedges, labels, title="Animal sex", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title("sex_upon_outcome distribution")
    plt.show()
    '''
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    types = d['outcome_type'].unique()
    width = 0.15
    N = len(d['sex_upon_outcome'].unique())
    ind = np.arange(10)
    print(ind)
    fig, ax = plt.subplots()
    rects = []
    for i, t in enumerate(d['sex_upon_outcome'].unique()):
        data = d.loc[d['sex_upon_outcome'] == t, ['sex_upon_outcome', 'outcome_type']]
        bar_heights = []

        for j in types:
            data_subset = data.loc[data['outcome_type'] == j, 'outcome_type']
            bar_heights.append(len(data_subset))

        rects.append(ax.bar(ind + (i * width), bar_heights, width, color=colours[i]))

    ax.set_ylabel("No. instances")
    ax.set_title("Graph showing outcome for each animal sex")
    ax.set_xticks(ind + width * 2.5)
    ax.set_xticklabels(types)
    ax.legend(rects, d['sex_upon_outcome'].unique())
    plt.show()


#age_analysis(data)
#outcome_subtype_analysis(data)
#type_analysis(data)
#name_analysis(data)
sex_analysis(data)
