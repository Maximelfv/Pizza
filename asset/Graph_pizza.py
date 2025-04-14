import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from read_data import data_pizzas
from code_pizza import moyenne_ca_pizza

# 1️⃣ Histogramme : Distribution du chiffre d'affaires total par commande
"""
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
"""
####################################################################################
####################################################################################
####################################################################################


############################### PARTIE IA ##########################################

from code_pizza import y_train, y_test, X_train, X_test, y_pred, model, mse, r2

import matplotlib.pyplot as plt

# 1️⃣ Courbe : Prédictions vs Réalité
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Valeurs réelles', linestyle='-', marker='o', color='blue')
plt.plot(y_pred, label='Prédictions', linestyle='--', marker='x', color='orange')

plt.title("📈 Évolution du Chiffre d'Affaires - Réel vs Prédit")
plt.xlabel("Date de Commande")
plt.ylabel("Chiffre d'Affaires (€)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Ajout des métriques sur le graphique
plt.text(0.01, 0.95, f"MSE : {mse:.2f}\nR² : {r2:.2f}", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black'), fontsize=10)

plt.tight_layout()
plt.show()


# 2️⃣ Nuage de points (Scatter) : Réel vs Prédit
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.7, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Prédiction parfaite')

plt.title("🎯 Prédictions vs Réalité - Chiffre d'Affaires")
plt.xlabel("Valeurs Réelles du CA (€)")
plt.ylabel("Valeurs Prédites du CA (€)")
plt.legend()
plt.grid(True)

# Ajout des métriques sur le scatter plot
plt.text(0.05, 0.85, f"MSE : {mse:.2f}\nR² : {r2:.2f}", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black'), fontsize=10)

plt.tight_layout()
plt.show()

