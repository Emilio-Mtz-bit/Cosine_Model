import numpy as np
import pandas as pd

data = pd.read_csv("../data/datos_grasas_Tec.csv", encoding="latin-1")

for i in data.columns:
    na = data[i].isnull().sum()
    if na > 45:
        data.drop(columns=i,inplace=True)
        
        
data.info()

data.to_csv("../data/datos_grasas_Tec_limpio.csv", index=False)