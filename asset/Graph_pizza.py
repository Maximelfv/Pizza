import matplotlib.pyplot as plt

from read_data import data_pizzas
from code_pizza import moyenne_ca_pizza

# 1️⃣ Histogramme : Distribution du chiffre d'affaires total par commande

# Distribution du CA par Pizza
plt.figure(figsize=(10,5))
plt.hist(data_pizzas['Chiffre Affaire'], bins=30, color='orange', edgecolor='black')
plt.title("Distribution du Chiffre d'Affaires par Pizza")
plt.xlabel("Chiffre d'Affaires (€)")
plt.ylabel("Nombre de Pizzas Vendues")
plt.grid(True)
plt.show()



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data_pizzas[data_pizzas['Chiffre Affaire'] > moyenne_ca_pizza]['Chiffre Affaire'], bins=15, color='red', edgecolor='black')
plt.title("Pizzas avec CA > Moyenne")
plt.xlabel("Chiffre d'Affaires (€)")
plt.ylabel("Nombre de Pizzas")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(data_pizzas[data_pizzas['Chiffre Affaire'] <= moyenne_ca_pizza]['Chiffre Affaire'], bins=15, color='yellow', edgecolor='black')
plt.title("Pizzas avec CA ≤ Moyenne")
plt.xlabel("Chiffre d'Affaires (€)")
plt.ylabel("Nombre de Pizzas")
plt.grid(True)

plt.subplots_adjust(wspace=0.4)
plt.show()


# 2️⃣ Graphique en barres : Chiffre d'affaires par pizza

# Calcul du chiffre d'affaires par pizza

# CA par Pizza
plt.figure(figsize=(12, 6))
plt.bar(data_pizzas['Name'], data_pizzas['Chiffre Affaire'], color='orange')
plt.xticks(
    ticks=np.arange(0, len(data_pizzas['Name']), 4),
    labels=data_pizzas['Name'][::4],
    rotation=90
)
plt.title("Chiffre d'Affaires par Pizza")
plt.xlabel("Pizza")
plt.ylabel("Chiffre d'Affaires (€)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# 3️⃣ Nuage de points (scatter plot) : Relation entre prix unitaire et quantité vendue

# Pizzas : Prix unitaire vs Quantité vendue
plt.figure(figsize=(8,5))
plt.scatter(data_pizzas['Prix unitaire'], data_pizzas['Quantité total commande'], color='green')
plt.xlabel('Prix Unitaire (€)')
plt.ylabel('Quantité Totale Commandée')
plt.title("Relation Prix Unitaire / Quantité (Pizzas)")
plt.grid(True)
plt.show()

