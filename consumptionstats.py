import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
input=sys.argv[1]
data =  pd.read_csv(".\\food-data.csv")
data['wastage%']=0
for i in range(len(data)): #iterating between all the rows of dataframe
    data['wastage%'][i] = ((data['served-qty'][i] - data['consumed-qty'][i])/data['served-qty'][i])*100


# df = data.to_html()
# print(df)
apr_3rd = data[data["date"] == input]
fig = px.line(apr_3rd, x = 'menu', y = 'wastage%', title=input+' wastage')
#fig.show()
#fig.write_image("monday.png") 
fig.write_html(input+".html")

def display ():
    

    with open(input+'.html') as f:
        print(f.read())

display()