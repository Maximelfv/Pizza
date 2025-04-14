import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from read_data import data_clients,info,df

########################### CLIENT   ########################################################
#  Calcul moyenne, médianne, max et min du chiffre d'affaires des clients
moyenne_ca_client, mediane_ca_client, max_ca_client, min_ca_client = info(data_clients['Prix total'])

# Calcul moyenne, médianne, max et min des quantités de pizzas commandées
moyenne_qte_client, mediane_qte_client, max_qte_client, min_qte_client = info(data_clients['Quantite de pizza'])

##  Calcul des clients dont chiffres d'affaires > 4*moyenne
clients_plus_rentables = data_clients[data_clients['Prix total'] > 4*moyenne_ca_client]
nb_clients_rentable = len(clients_plus_rentables)

# Calcul de la correlation entre le prix total et la quantité de pizza commandée
corr_client = data_clients['Prix total'].corr(data_clients['Quantite de pizza'])

# Recherche du client le plus depensier
client_plus_depensier = data_clients[data_clients['Prix total'] == max(data_clients['Prix total'])]['client']

# Calcul de l'écart-type pour le CA des clients
std_ca_client = np.std(data_clients['Prix total'])

# Détection des clients aberrants (> 2 écarts-types)
aberrants_clients_sup = data_clients[data_clients['Prix total'] > (moyenne_ca_client + 2 * std_ca_client)]
aberrants_clients_inf = data_clients[data_clients['Prix total'] < (moyenne_ca_client - 2 * std_ca_client)]


# Détection des client aberrants plus stricte (> 8 écarts-types)
aberrants_clients_sup_v2 = data_clients[data_clients['Prix total'] > (moyenne_ca_client + 8 * std_ca_client)]
aberrants_clients_inf_v2 = data_clients[data_clients['Prix total'] < (moyenne_ca_client - 8 * std_ca_client)]



####################################################################################
####################################################################################
####################################################################################


############################### PARTIE IA ##########################################
# ============================= KMeans ====================================
from sklearn.cluster import KMeans

X = data_clients[['Prix total', 'Quantite de pizza']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
data_clients['Cluster'] = kmeans.fit_predict(X)
# Cela va regrouper les clients en 3 clusters basés sur le prix total et la quantité de pizza commandée
# Afin de classer les clients en fonction de leur comportement d'achat


# ============================= Régression Linéaire =========================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.arange(len(data_clients)).reshape(-1, 1) # J'utilise pas data_clients['client'] car c'est pas numérique
y = data_clients['Quantite de pizza'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred) # Coefficient de détermination R²



# Prédiction de la quantité de pizza pour le client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Séparation en jeu d'entraînement et de test (75% / 25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Entraînement du modèle
model_v2 = LinearRegression()
model_v2.fit(X_train, y_train)

# Prédictions
y_pred_v2 = model_v2.predict(X_test)

# Évaluation
r2_v2 = r2_score(y_test, y_pred_v2)
mse = mean_squared_error(y_test, y_pred_v2)
mae = mean_absolute_error(y_test, y_pred_v2)




# Prediction sur le nombre de client sur les 3 derniers mois
data_clients['Mois'] = data_clients['Horodatage'].dt.month

# 👥 Compter le nombre de clients uniques par mois
clients_par_mois = data_clients.groupby('Mois')['client'].nunique().reset_index()
clients_par_mois.rename(columns={'client': 'Nombre_Clients'}, inplace=True)

# 🎯 Utiliser les 9 premiers mois comme ensemble d'entraînement
train = clients_par_mois[clients_par_mois['Mois'] <= 9]
test = clients_par_mois[clients_par_mois['Mois'] > 9].copy()

# 📈 Régression linéaire
X_train_3mois = train[['Mois']]
y_train_3mois = train['Nombre_Clients']

X_test_3mois = test[['Mois']]  

model_3mois = LinearRegression()
model_3mois.fit(X_train_3mois, y_train_3mois)

# 🔮 Prédiction
y_pred_3mois = model_3mois.predict(X_test_3mois)

# Évaluation
r2_3mois = r2_score(test['Nombre_Clients'], y_pred_3mois)
mse_3mois = mean_squared_error(test['Nombre_Clients'], y_pred_3mois)
mae_3mois = mean_absolute_error(test['Nombre_Clients'], y_pred_3mois)

# 📊 Résultats
test.loc[:, 'Prévision_Clients'] = y_pred_3mois.astype(int)




# ============================== Pareto ==================================
data_pareto = data_clients.sort_values(by='Prix total', ascending=False)
data_pareto['Cumsum'] = data_pareto['Prix total'].cumsum()
data_pareto['Cumsum_pct'] = 100 * data_pareto['Cumsum'] / data_pareto['Prix total'].sum()
seuil_80 = data_pareto[data_pareto['Cumsum_pct'] <= 80]



# ============================== Analyse RFM ========================
rfm = data_clients.copy()
rfm['Récence'] = (pd.Timestamp("2024-12-31") - rfm['Horodatage']).dt.days
rfm.rename(columns={'Prix total': 'Montant', 'client': 'ID'}, inplace=True)
rfm['Fréquence'] = 1
rfm = rfm.groupby('ID').agg({
    'Récence': 'min',
    'Fréquence': 'sum',
    'Montant': 'sum'
}).reset_index()




###########################################################################################
###########################################################################################
###########################################################################################

# ===================================== Main ==============================================

def __main__():
    print(" =============================================================")
    print(" =============================================================")
    print("              Partie Client                   ")
    print(" Voici les données clients :")
    print(f" Moyenne CA: {moyenne_ca_client}\n Médiane CA: {mediane_ca_client},\n Max CA: {max_ca_client},\n Min CA: {min_ca_client}\n")
    print(f" Moyenne Qte: {moyenne_qte_client}\n Médiane Qte: {mediane_qte_client},\n Max Qte: {max_qte_client},\n Min Qte: {min_qte_client}\n")
    print(f" Les clients les plus rentables :\n {clients_plus_rentables} \n Soit {nb_clients_rentable} clients dont le CA est supérieur à {4*moyenne_ca_client:.2f}")
    print(f"Corrélation entre total dépensé et quantité commandée (Clients) : {corr_client:.2f}")
    print(f"Voici le client qui a dépensé le plus : {client_plus_depensier.values[0]} avec {max(data_clients['Prix total'])} €")
    print(f"Valeur de l'écart-type pour les clients : {std_ca_client:.2f}")
    print(f"Nombre de clients avec CA aberrant (> 2 écarts-types) : {len(aberrants_clients_sup)}")
    print(f"Nombre de clients avec CA aberrant (< 2 écarts-types) : {len(aberrants_clients_inf)}")
    print(f"Nombre total de clients : {len(data_clients)}")
    print(f"Nombre de clients avec CA aberrant (> 8 écarts-types) : {len(aberrants_clients_sup_v2)}")
    print(f"Nombre de clients avec CA aberrant (< 8 écarts-types) : {len(aberrants_clients_inf_v2)}")
    print(aberrants_clients_sup_v2[['client', 'Prix total']])
    print(aberrants_clients_inf_v2[['client', 'Prix total']])
    print(" =============================================================")
    print(" =============================================================")
    print("              Partie IA                   ")
    print(" KMeans :")
    print(data_clients[['client', 'Cluster']])
    print(" Regression Linéaire :")
    print(f" Le coefficient de détermination R² est : {r2:.2f}")
    print(" Regression linéaire prédiction :")
    print(f"R² score (test set) : {r2_v2:.2f}")
    print(f"MSE (test set)      : {mse:.2f}")
    print(f"MAE (test set)      : {mae:.2f}")
    print(" Voici les données de prévision des client sur les 3 derniers mois :")
    print(f"R² score (test set) : {r2_3mois:.2f}")
    print(f"MSE (test set)      : {mse_3mois:.2f}")
    print(f"MAE (test set)      : {mae_3mois:.2f}")
    print("Prédiction du nombre de clients pour les 3 derniers mois :")
    print(test[['Mois', 'Nombre_Clients', 'Prévision_Clients']])
    print(" Pareto :")
    print(" Voici les nombre de clients qui représentent 80% du CA :")
    print(f"{len(seuil_80)} clients représentent 80% du CA. sur {len(data_clients)} clients")
    print(" RFM :")
    print(rfm[['ID', 'Récence', 'Fréquence', 'Montant']])
    print(" Voici les 10 clients les plus récents :")
    print(rfm.sort_values(by='Récence').head(10))
    print(" Voici les 10 clients les plus fréquents :")
    print(rfm.sort_values(by='Fréquence', ascending=False).head(10))
    print(" Voici les 10 clients les plus dépensiers :")
    print(rfm.sort_values(by='Montant', ascending=False).head(10))
    print(" =============================================================")



if __name__ == "__main__":
    __main__()