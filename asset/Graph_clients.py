import matplotlib.pyplot as plt

from read_data import data_clients
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
    ticks=np.arange(0, len(data_clients['client']), 50),  # Un tick sur 50 clients
    labels=data_clients['client'][::50],
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
from code_client import data_clients

plt.scatter(data_clients['Prix total'], data_clients['Quantite de pizza'], c=data_clients['Cluster'], cmap='rainbow')
plt.xlabel('Prix total (€)')
plt.ylabel('Quantité de pizza')
plt.title('Classification des Produits par Qualité')
plt.grid(True)
plt.show()


from code_client import X, y, y_pred, data_pareto


# Graphiques
plt.figure(figsize=(12, 5))

# 1️⃣ Régression linéaire
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Données') 
plt.plot(X, y_pred, color='red', label='Régression linéaire')
plt.title("Quantité de pizza par client (indexé)")
plt.xlabel("Index client")
plt.ylabel("Quantité de pizza")
plt.legend()
plt.grid(True)

# 2️⃣ Courbe de Pareto
plt.subplot(1, 2, 2)
plt.plot(data_pareto['Cumsum_pct'].values, color='green')
plt.axhline(y=80, color='red', linestyle='--', label='Seuil 80%')
plt.title("Courbe de Pareto - CA cumulé")
plt.xlabel("Clients (triés par CA)")
plt.ylabel("Pourcentage cumulé du CA")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()






