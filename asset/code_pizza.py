import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from read_data import data_pizzas,info,df

#  Calcul moyenne, médianne, max et min du chiffre d'affaires des pizzas 
moyenne_ca_pizza,mediane_ca_pizza,max_ca_pizza,min_ca_pizza = info(data_pizzas['Chiffre Affaire'])

#  Affichage des pizzas dont chiffres d'affaires > 3*moyenne
pizzas_plus_rentables = data_pizzas[data_pizzas['Chiffre Affaire'] > 3*moyenne_ca_pizza]
nb_pizzas_rentable = len(pizzas_plus_rentables)
print(f" Les pizzas les plus rentables :\n {pizzas_plus_rentables} \n Soit {nb_pizzas_rentable} pizzass dont le CA est supérieur à {3*moyenne_ca_pizza:.2f}")

# Pizza la plus chère
prix_max = max(df['unit_price'])
name_pizza_max = df[df['unit_price'] == prix_max]['pizza_name'].values 
print(f" Voici la pizza la plus chère : {name_pizza_max[0]} à {prix_max} €")

# Calcul du coefficient de corrélation entre prix unitaire et quantité vendue des pizzas
correlation = data_pizzas['Prix unitaire'].corr(data_pizzas['Quantité total commande'])
print(f" Voici la corrélation entre le prix de chaque pizza et la quantité : {correlation}")
# corr = -0.0033... proche de 0 donc il n'y a aucun lien entre le prix des pizzas et la quantité vendu

#Recherche de la pizza la plus rentable
pizza_plus_rentable = data_pizzas[data_pizzas['Chiffre Affaire']==max(data_pizzas['Chiffre Affaire'])]['Name']
print(f" Voici la pizza la plus rentable : {pizza_plus_rentable}")



########################### PIZZA  ###########################################################


std_ca_pizza = np.std(data_pizzas['Chiffre Affaire'])

# Détection des pizzas aberrantes (2 fois l'écart-type)
aberrantes_pizzas_sup = data_pizzas[data_pizzas['Chiffre Affaire'] > (moyenne_ca_pizza + 2 * std_ca_pizza)]
aberrantes_pizzas_inf = data_pizzas[data_pizzas['Chiffre Affaire'] < (moyenne_ca_pizza - 2 * std_ca_pizza)]

print(f"Valeur de l'écart-type pour les pizzas : {std_ca_pizza:.2f}")
print(f"Nombre de pizzas dont le chiffre d'affaires est aberrant (> 2 écarts-types) : {len(aberrantes_pizzas_sup)}")
print(f"Nombre de pizzas dont le chiffre d'affaires est aberrant (< 2 écarts-types) : {len(aberrantes_pizzas_inf)}")
print(f"Nombre total de pizzas uniques : {len(data_pizzas['Name'])}")
print(aberrantes_pizzas_sup[['Name', 'Chiffre Affaire']])
print(aberrantes_pizzas_inf[['Name', 'Chiffre Affaire']])

# Détection plus stricte (8 fois l'écart-type)
aberrantes_pizzas_sup_v2 = data_pizzas[data_pizzas['Chiffre Affaire'] > (moyenne_ca_pizza + 8 * std_ca_pizza)] \
                        .drop_duplicates(subset='Name') \
                        .sort_values(by='Chiffre Affaire', ascending=True)
                        
aberrantes_pizzas_inf_v2 = data_pizzas[data_pizzas['Chiffre Affaire'] < (moyenne_ca_pizza - 8 * std_ca_pizza)] \
                        .drop_duplicates(subset='Name') \
                        .sort_values(by='Chiffre Affaire', ascending=True)

print(f"Nombre de pizzas dont le chiffre d'affaires est aberrant (> 8 écarts-types) : {len(aberrantes_pizzas_sup_v2)}")
print(f"Nombre de pizzas dont le chiffre d'affaires est aberrant (< 8 écarts-types) : {len(aberrantes_pizzas_inf_v2)}")
print(aberrantes_pizzas_sup_v2[['Name', 'Chiffre Affaire']])
print(aberrantes_pizzas_inf_v2[['Name', 'Chiffre Affaire']])

####################################################################################
####################################################################################
####################################################################################


############################### PARTIE IA ##########################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = data_pizzas[['Horodatage']]
X['Horodatage'] = pd.to_datetime(X['Horodatage'])
X['Horodatage'] = X['Horodatage'].map(pd.Timestamp.toordinal)  # conversion en entier

y = data_pizzas['Chiffre Affaire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ✅ Évaluation du modèle (régression)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
print(f"Erreur absolue moyenne (MAE)     : {mae:.2f}")
print(f"Coefficient de détermination (R²): {r2:.2f}")
















