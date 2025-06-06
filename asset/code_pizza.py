import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from read_data import data_pizzas,info,df,data_pizzas_mois

########################### PIZZA   ########################################################
#  Calcul moyenne, médianne, max et min du chiffre d'affaires des pizzas 
moyenne_ca_pizza,mediane_ca_pizza,max_ca_pizza,min_ca_pizza = info(data_pizzas['Chiffre Affaire'])

# Calcul moyenne, médianne, max et min des quantités de pizzas commandées
moyenne_qte_pizza,mediane_qte_pizza,max_qte_pizza,min_qte_pizza = info(data_pizzas['Quantité total commande'])

#  Affichage des pizzas dont chiffres d'affaires > 3*moyenne
pizzas_plus_rentables = data_pizzas[data_pizzas['Chiffre Affaire'] > 3*moyenne_ca_pizza]
nb_pizzas_rentable = len(pizzas_plus_rentables)

# Pizza la plus chère
prix_max = max(df['unit_price'])
name_pizza_max = df[df['unit_price'] == prix_max]['pizza_name'].values 

# Calcul du coefficient de corrélation entre prix unitaire et quantité vendue des pizzas
correlation = data_pizzas['Prix unitaire'].corr(data_pizzas['Quantité total commande'])

#Recherche de la pizza la plus rentable
pizza_plus_rentable = data_pizzas[data_pizzas['Chiffre Affaire']==max(data_pizzas['Chiffre Affaire'])]['Name']

# Calcul de l'écart-type pour le CA des pizzas
std_ca_pizza = np.std(data_pizzas['Chiffre Affaire'])

# Détection des pizzas aberrantes (2 fois l'écart-type)
aberrantes_pizzas_sup = data_pizzas[data_pizzas['Chiffre Affaire'] > (moyenne_ca_pizza + 2 * std_ca_pizza)]
aberrantes_pizzas_inf = data_pizzas[data_pizzas['Chiffre Affaire'] < (moyenne_ca_pizza - 2 * std_ca_pizza)]

# Détection plus stricte (8 fois l'écart-type)
aberrantes_pizzas_sup_v2 = data_pizzas[data_pizzas['Chiffre Affaire'] > (moyenne_ca_pizza + 8 * std_ca_pizza)] \
                        .drop_duplicates(subset='Name') \
                        .sort_values(by='Chiffre Affaire', ascending=True)
                        
aberrantes_pizzas_inf_v2 = data_pizzas[data_pizzas['Chiffre Affaire'] < (moyenne_ca_pizza - 8 * std_ca_pizza)] \
                        .drop_duplicates(subset='Name') \
                        .sort_values(by='Chiffre Affaire', ascending=True)


####################################################################################
####################################################################################
####################################################################################

############################### PARTIE IA ##########################################
# =========== Régression linéaire + Prédiction sur les 3 dernier mois  =============
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = data_pizzas_mois[['Mois']]
y = data_pizzas_mois['Chiffre Affaire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ✅ Évaluation du modèle (régression)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ============== Régrssion linéaire + Prédiction sur les 13 dernières semaines =============
from read_data import df_semaine

X_semaine = df_semaine[["total_price"]]
y_semaine = df_semaine["quantity"]

X_train_semaine, X_test_semaine, y_train_semaine, y_test_semaine = train_test_split(X_semaine, y_semaine, test_size=0.25, random_state=42)

modele_semaine = LinearRegression()
modele_semaine.fit(X_train_semaine, y_train_semaine)

y_pred_semaine = modele_semaine.predict(X_test_semaine)
r2_semaine = r2_score(y_test_semaine, y_pred_semaine) 
mse_semaine = mean_squared_error(y_test_semaine, y_pred_semaine)
mae_semaine = mean_absolute_error(y_test_semaine, y_pred_semaine)





# ===================== Prédiction journalière  =========================
from read_data import df_journalier

X_jour = df_journalier[["total_price"]]
y_jour = df_journalier["quantity"] 


X_train_jour, X_test_jour, y_train_jour, y_test_jour = train_test_split(X_jour, y_jour, test_size=0.25, random_state=42)

modele_jour= LinearRegression()
modele_jour.fit(X_train_jour, y_train_jour)


y_pred_jour = modele_jour.predict(X_test_jour)
score_precision = r2_score(y_test_jour, y_pred_jour)
erreur_quadratique = mean_squared_error(y_test_jour, y_pred_jour)


#Exemple de prédiction

prediction=modele_jour.predict(pd.DataFrame([[36]], columns=["total_price"]))


modele_jour.fit(X_jour, y_jour)
df_journalier["prediction"] = modele_jour.predict(X_jour)
df_journalier["prediction"] = df_journalier["prediction"].round().astype(int)
df_journalier["quantity"] = df_journalier["quantity"].round().astype(int)







def __main__():
    print(" =============================================================")
    print(" =============================================================")
    print("              Partie pizza                   ")
    print(" Voici les données pizzas :")
    print(f" Moyenne CA: {moyenne_ca_pizza}\n Médiane CA: {mediane_ca_pizza},\n Max CA: {max_ca_pizza},\n Min CA: {min_ca_pizza}\n")
    print(" Voici les données sur les quantités :")
    print(f" Moyenne CA: {moyenne_qte_pizza}\n Médiane CA: {mediane_qte_pizza},\n Max CA: {max_qte_pizza},\n Min CA: {min_qte_pizza}\n")
    print(f" Les pizzas les plus rentables :\n {pizzas_plus_rentables} \n Soit {nb_pizzas_rentable} pizzass dont le CA est supérieur à {3*moyenne_ca_pizza:.2f}")
    print(f" Voici la pizza la plus chère : {name_pizza_max[0]} à {prix_max} €")
    print(f" Voici la corrélation entre le prix de chaque pizza et la quantité : {correlation}")    
    # corr = -0.0033... proche de 0 donc il n'y a aucun lien entre le prix des pizzas et la quantité vendu
    print(f" Voici la pizza la plus rentable : {pizza_plus_rentable}")
    print(f"Valeur de l'écart-type pour les pizzas : {std_ca_pizza:.2f}")
    print(f"Nombre de pizzas dont le chiffre d'affaires est aberrant (> 2 écarts-types) : {len(aberrantes_pizzas_sup)}")
    print(f"Nombre de pizzas dont le chiffre d'affaires est aberrant (< 2 écarts-types) : {len(aberrantes_pizzas_inf)}")
    print(f"Nombre total de pizzas uniques : {len(data_pizzas['Name'])}")
    print(aberrantes_pizzas_sup[['Name', 'Chiffre Affaire']])
    print(aberrantes_pizzas_inf[['Name', 'Chiffre Affaire']])
    print(f"Nombre de pizzas dont le chiffre d'affaires est aberrant (> 8 écarts-types) : {len(aberrantes_pizzas_sup_v2)}")
    print(f"Nombre de pizzas dont le chiffre d'affaires est aberrant (< 8 écarts-types) : {len(aberrantes_pizzas_inf_v2)}")
    print(aberrantes_pizzas_sup_v2[['Name', 'Chiffre Affaire']])
    print(aberrantes_pizzas_inf_v2[['Name', 'Chiffre Affaire']])
    print(" =============================================================")
    print(" =============================================================")
    print("              Partie IA                   ")
    print(" Régression linéaire avec prédiction sur les 3 derniers mois :")
    print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
    print(f"Erreur absolue moyenne (MAE)     : {mae:.2f}")
    print(f"Coefficient de détermination (R²): {r2:.2f}")
    print("\n Régression linéaire avec prédiction sur les 13 dernières semaines :")
    print(f"Erreur quadratique moyenne (MSE) : {mse_semaine:.2f}")
    print(f"Erreur absolue moyenne (MAE)     : {mae_semaine:.2f}")
    print(f"Coefficient de détermination (R²): {r2_semaine:.2f}")
    print("==========================")
    print(" Régression linéaire journalière :")
    print("Score de précision :", round(score_precision,3))
    print("Erreur quadratique moyenne :", round(erreur_quadratique,5))
    print("\n Nombre de pizzas prédit :",round(prediction[0]))
    print("\n Comparatif réel vs prédit (par jour) :")
    print(df_journalier[["order_date", "quantity", "prediction"]].sort_values("order_date").head(11))






if __name__ == "__main__":
    __main__()