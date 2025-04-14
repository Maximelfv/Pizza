import matplotlib.pyplot as plt
import numpy as np

from code_client import data_clients
"""
# 1️⃣ Histogramme : Distribution du chiffre d'affaires total par commande

# Distribution du CA par Client
plt.figure(figsize=(10,5))
plt.hist(data_clients['Prix total'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution des Dépenses par Client")
plt.xlabel("Total Dépensé par Client (€)")
plt.ylabel("Nombre de pizzas achetées")
plt.grid(True)
plt.show()



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data_clients[data_clients['Prix total'] > 90]['Prix total'], bins=15, color='red', edgecolor='black')
plt.title("Dépenses Clients > 90 €")
plt.xlabel("Total Dépensé (€)")
plt.ylabel("Nombre de Clients")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(data_clients[data_clients['Prix total'] <= 90]['Prix total'], bins=15, color='yellow', edgecolor='black')
plt.title("Dépenses Clients ≤ 90 €")
plt.xlabel("Total Dépensé (€)")
plt.ylabel("Nombre de Clients")
plt.grid(True)

plt.subplots_adjust(wspace=0.4)
plt.show()





# 2️⃣ Graphique en barres : Chiffre d'affaires par pizza

# CA par Client
plt.figure(figsize=(12, 6))
plt.bar(data_clients['client'], data_clients['Prix total'], color='purple')
plt.xticks(
    ticks=np.arange(0, len(data_clients['client']), 1000),  # Un tick sur 1000 clients
    labels=data_clients['client'][::1000],
    rotation=90
)
plt.title("Total Dépensé par Client")
plt.xlabel("Client")
plt.ylabel("Total Dépensé (€)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()




# 3️⃣ Nuage de points (scatter plot) : Relation entre prix unitaire et quantité vendue

# Clients : Total dépensé vs Quantité de pizza
plt.figure(figsize=(8,5))
plt.scatter(data_clients['Prix total'], data_clients['Quantite de pizza'], color='purple')
plt.xlabel('Total Dépensé (€)')
plt.ylabel('Quantité Totale Commandée')
plt.title("Relation Dépenses / Quantité (Clients)")
plt.grid(True)
plt.show()

"""



####################################################################################
####################################################################################
####################################################################################


############################### PARTIE IA ##########################################
# ============================= KMeans ====================================
plt.scatter(data_clients['Prix total'], data_clients['Quantite de pizza'], c=data_clients['Cluster'], cmap='rainbow')
plt.xlabel('Prix total (€)')
plt.ylabel('Quantité de pizza')
plt.title('Classification des Produits par Qualité')
plt.grid(True)
plt.show()



# ============================= Régression Linéaire =========================
from code_client import X, y, y_pred, r2

# Graphiques
plt.figure(figsize=(12, 5))

plt.scatter(X, y, color='blue', label='Données') 
plt.plot(X, y_pred, color='red', label='Régression linéaire')
plt.title("Quantité de pizza par client (indexé)")
plt.xlabel("Index client")
plt.ylabel("Quantité de pizza")
plt.text(0.5, 0.9, f'R² = {r2:.2f}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
plt.legend()
plt.grid(True)


# Prédiction de la quantité de pizza pour le client
from code_client import y_test, y_pred_v2, r2_v2

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Valeurs réelles', marker='o')
plt.plot(y_pred_v2, label='Prédictions', marker='x')
plt.title(" Régression Linéaire - Quantité de pizza")
plt.xlabel("Index (test)")
plt.ylabel("Quantité de pizza")
plt.text(0.5, 0.9, f'R² = {r2_v2:.2f}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Prediction sur le nombre de client sur les 3 derniers mois
from code_client import train, test, y_train_3mois, y_pred_3mois

plt.figure(figsize=(10,6))
plt.plot(train['Mois'], y_train_3mois, marker='o', label='Historique (mois 1-9)')
plt.plot(test['Mois'], test['Nombre_Clients'], marker='x', label='Réel (mois 10-12)')
plt.plot(test['Mois'], y_pred_3mois, linestyle='--', marker='s', label='Prévu (mois 10-12)')
plt.title(" Prédiction du nombre de clients pour les 3 derniers mois")
plt.xlabel("Mois")
plt.ylabel("Nombre de Clients")
plt.xticks(range(1,13))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# Prediction sur le nombre de client sur les 13 dernières semaines
from code_client import train_semaine, test_semaine, y_train_semaine, y_pred_semaine

plt.figure(figsize=(10, 6))

plt.plot(train_semaine['Semaine'], y_train_semaine, marker='o', label='Historique (semaines 1-38)', color='blue')
plt.plot(test_semaine['Semaine'], test_semaine['Nombre_Clients'], marker='x', label='Réel (semaines 39-52)', color='green')
plt.plot(test_semaine['Semaine'], y_pred_semaine, linestyle='--', marker='s', label='Prévu (semaines 39-52)', color='orange')
plt.title(" Prédiction du nombre de clients pour les 13 dernières semaines")
plt.xlabel("Numéro de Semaine")
plt.ylabel("Nombre de Clients")
plt.xticks(test_semaine['Semaine'].tolist())
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ================================== Pareto ==================================
from code_client import data_pareto

plt.figure(figsize=(12, 5))

plt.plot(data_pareto['Cumsum_pct'].values, color='green')
plt.axhline(y=80, color='red', linestyle='--', label='Seuil 80%')
plt.title("Courbe de Pareto - CA cumulé")
plt.xlabel("Clients (triés par CA)")
plt.ylabel("Pourcentage cumulé du CA")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()






