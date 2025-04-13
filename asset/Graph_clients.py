import matplotlib.pyplot as plt

from read_data import data_clients

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
