import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

####################################################################################
####################################################################################
####################################################################################


############################### PARTIE IA ##########################################
# =========== Régression linéaire + Prédiction sur les 3 dernier mois  =============
from code_pizza import y_train, y_test, X_train, X_test, y_pred, model, mse, r2, mae


# 1️⃣ Courbe : Prédictions vs Réalité
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Valeurs réelles', linestyle='-', marker='o', color='blue')
plt.plot(y_pred, label='Prédictions', linestyle='--', marker='x', color='orange')

plt.title(" Évolution du Chiffre d'Affaires par mois - Réel vs Prédit")
plt.xlabel("Date de Commande")
plt.ylabel("Chiffre d'Affaires (€)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.text(0.01, 0.95, f"MSE : {mse:.2f}\nR² : {r2:.2f}", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black'), fontsize=10)

plt.tight_layout()
plt.show()


# 2️⃣ Nuage de points (Scatter) : Réel vs Prédit
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.7, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Prédiction parfaite')

plt.title(" Prédictions vs Réalité - Chiffre d'Affaires par mois ")
plt.xlabel("Valeurs Réelles du CA (€)")
plt.ylabel("Valeurs Prédites du CA (€)")
plt.legend()
plt.grid(True)
plt.text(0.05, 0.85, f"MSE : {mse:.2f}\nR² : {r2:.2f}", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black'), fontsize=10)

plt.tight_layout()
plt.show()


# ============== Régrssion linéaire + Prédiction sur les 13 dernières semaines =============
from code_pizza import y_test_semaine, y_pred_semaine, X_test_semaine, y_test_semaine, y_pred_semaine, modele_semaine, mse_semaine, r2_semaine, mae_semaine

plt.figure(figsize=(12, 6))
plt.plot(y_test_semaine.values, label='Valeurs réelles', linestyle='-', marker='o', color='blue')
plt.plot(y_pred_semaine, label='Prédictions', linestyle='--', marker='x', color='orange')
plt.title(" Évolution du Chiffre d'Affaires par semaine - Réel vs Prédit")
plt.xlabel("Date de Commande")
plt.ylabel("Chiffre d'Affaires (€)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.text(0.01, 0.95, f"MSE : {mse_semaine:.2f}\nR² : {r2_semaine:.2f}", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black'), fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_semaine, y_pred_semaine, color='green', alpha=0.7, edgecolor='k')
plt.plot([y_test_semaine.min(), y_test_semaine.max()], [y_test_semaine.min(), y_test_semaine.max()], 'r--', label='Prédiction parfaite')
plt.title(" Prédictions vs Réalité - Chiffre d'Affaires par semaine ")  
plt.xlabel("Valeurs Réelles du CA (€)")
plt.ylabel("Valeurs Prédites du CA (€)")
plt.legend()
plt.grid(True)
plt.text(0.05, 0.85, f"MSE : {mse_semaine:.2f}\nR² : {r2_semaine:.2f}", transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black'), fontsize=10)
plt.tight_layout()
plt.show()


# =================== Régression linéaire avec prédiction journalière ======================
from code_pizza import df_journalier, y_test_jour, y_pred_jour

plt.figure(figsize=(8, 6))
plt.scatter(y_test_jour, y_pred_jour, alpha=0.5)
plt.plot([y_test_jour.min(), y_test_jour.max()], [y_test_jour.min(), y_test_jour.max()], 'r--')
plt.xlabel("Vraie quantité de pizzas vendues")
plt.ylabel("Quantité prédite")
plt.title("Quantité réelle vs prédite par jour ")
plt.grid(True)
plt.tight_layout()
plt.show()
