import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../data/datos_grasas_Tec_limpio.csv")
data = data.to_numpy()
data.shape

def Cos_Model(matriz_datos,fichas):
    matriz_datos_np = matriz_datos.to_numpy()
    
    for i in fichas:
        simil = cosine_similarity(i, matriz_datos_np)
        max_valor = simil.max()
        max_index = simil.argmax()


        print("----- Similaridad con productos interlub -----")
        print(f"Vector: {i}")
        for j in simil:
            print(f"Similitud con {matriz_datos.iloc[j,0]}: {j}")
            
        print(f"Best match for vector {i}: {max_valor}, idx: {max_index}")
            
        