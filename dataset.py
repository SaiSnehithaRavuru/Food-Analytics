import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data =  pd.read_csv(".\\food-data.csv")

html = data.to_html()
print(html)