import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../data/datos_grasas_Tec_limpio.csv")
data = data.select_dtypes(include=np.number)
data = data.drop(columns=["idDatosGrasas", "Registro NSF"])
data = data.fillna(0)
data = data.to_numpy()

lista = [[1, 460, 620, 0.31, np.nan, np.nan, np.nan, np.nan],
         [2, 460, 620, 0.31, np.nan, np.nan, np.nan, np.nan],
         [np.nan, 95, 408, np.nan, np.nan, np.nan, -20, 160], 
         [np.nan, 226, 315, 0.45, 107, np.nan, -40, 132],
         [np.nan, 226, 315, 0.45, 107, np.nan, -40, 132],
         [np.nan, 226, 315, 0.45, 107, np.nan, -40, 177],
         [np.nan, 226, 315, 0.45, 107, np.nan, -40, 177],
         [np.nan, 220, 400, 0.43, 72, np.nan, -40, 177],
         [np.nan, 220, 400, 0.43, 72, np.nan, -40, 177],
         [np.nan, 220, 315, 0.45, 50, np.nan, -40, 132],
         [np.nan, 220, 315, 0.45, 50, np.nan, -40, 177]]

def Cos_Model(matriz_datos, fichas):
    
    for i in fichas:
        vector_2d = np.array(i).reshape(1, -1)
        # Reemplazar NaN con 0 en el vector de entrada
        vector_2d = np.nan_to_num(vector_2d, nan=0.0)
        simil = cosine_similarity(vector_2d, matriz_datos)
        max_valor = simil.max()
        max_index = simil.argmax()

        print("----- Similaridad con productos interlub -----")
        print(f"Vector: {i}")
        
        for j in range(simil.shape[1]):
            print(f"Similitud con el producto en índice {j}: {simil[0, j]}")
            
        print(f"Best match for vector {i}: {max_valor}, idx: {max_index}")
            
Cos_Model(data, lista)