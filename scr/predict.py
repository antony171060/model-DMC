#!/usr/bin/env python
# coding: utf-8

# ## Lectura de Datos
# 


# Importamos la data
import pandas as pd


data = pd.read_csv('../data/raw/valid.csv')

X = data.drop("TARGET_XF", axis=1) #Feature Matrix 

# load the model from disk
filename = '../models/modelo_final.sav'
loaded_model = pickle.load(open(filename, 'rb'))

y_pred_test = loaded_model.predict(X)


y_pred_test
df_pred = pd.DataFrame(y_pred_test, columns=['Pred'])
df_pred


df_pred.to_csv('../data/scores/final_score.csv', index=False)

