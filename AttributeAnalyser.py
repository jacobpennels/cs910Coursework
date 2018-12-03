import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from project.DataCleaner import DataCleaner


data = pd.read_csv('../datafiles/aac_shelter_outcomes.csv')
dc = DataCleaner()
data = dc.clean_data(data)


