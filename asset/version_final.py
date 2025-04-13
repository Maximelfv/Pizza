# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 02:54:10 2025

@author: maxime.lefeuvre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./dataset/pizza_sales.csv")
print(df.head())
print(df.info())




#  Calcul moyenne, médianne, max et min


    
moyenne_ca_pizza,mediane_ca_pizza,max_ca_pizza,min_ca_pizza = info(data_pizzas['Chiffre Affaire'])
moyenne_ca_client,mediane_ca_client,max_ca_client,min_ca_client = info(data_clients['Prix total'])

#  Affichage des pizzas et des client dont chiffres d'affaires > 3*moyenne
pizzas_plus_rentables = data_pizzas[data_pizzas['Chiffre Affaire'] > 3*moyenne_ca_pizza]
nb_pizzas_rentable = len(pizzas_plus_rentables)
print(f" Les pizzas les plus rentables :\n {pizzas_plus_rentables} \n Soit {nb_pizzas_rentable} pizzass dont le CA est supérieur à {3*moyenne_ca_pizza:.2f}")

clients_plus_rentables = data_clients[data_clients['Prix total'] > 3*moyenne_ca_client]
nb_clients_rentable = len(clients_plus_rentables)
print(f" Les clients les plus rentables :\n {clients_plus_rentables} \n Soit {nb_clients_rentable} clients dont le CA est supérieur à {3*moyenne_ca_client:.2f}")



print("##########################################################################################")
print("##########################################################################################")
print("##########################################################################################")

# Pizza la plus chère
prix_max = max(df['unit_price'])
name_pizza_max = df[df['unit_price'] == prix_max]['pizza_name'].values

print(f" Voici la pizza la plus chère : {name_pizza_max[0]} à {prix_max} €")



# Calcul du coefficient de corrélation entre prix unitaire et quantité vendue des pizzas
correlation = data_pizzas['Prix unitaire'].corr(data_pizzas['Quantité total commande'])

print(f" Voici la corrélation entre le prix de chaque pizza et la quantité : {correlation}")

# corr = -0.0033... proche de 0 donc il n'y a aucun lien entre le prix des pizzas et la quantité vendu


corr_client = data_clients['Prix total'].corr(data_clients['Quantite de pizza'])
print(f"Corrélation entre total dépensé et quantité commandée (Clients) : {corr_client:.2f}")



pizza_plus_rentable = data_pizzas[data_pizzas['Chiffre Affaire']==max(data_pizzas['Chiffre Affaire'])]['Name']

print(f" Voici la pizza la plus rentable : {pizza_plus_rentable}")



client_plus_depensier = data_clients[data_clients['Prix total'] == max(data_clients['Prix total'])]['client']

print(f"Voici le client qui a dépensé le plus : {client_plus_depensier.values[0]} avec {max(data_clients['Prix total'])} €")



print("##########################################################################################")
print("##########################################################################################")
print("##########################################################################################")

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


########################### CLIENT   ########################################################


std_ca_client = np.std(data_clients['Prix total'])

# Détection des clients aberrants (> 2 écarts-types)
aberrants_clients_sup = data_clients[data_clients['Prix total'] > (moyenne_ca_client + 2 * std_ca_client)]
aberrants_clients_inf = data_clients[data_clients['Prix total'] < (moyenne_ca_client - 2 * std_ca_client)]

print(f"Valeur de l'écart-type pour les clients : {std_ca_client:.2f}")
print(f"Nombre de clients avec CA aberrant (> 2 écarts-types) : {len(aberrants_clients_sup)}")
print(f"Nombre de clients avec CA aberrant (< 2 écarts-types) : {len(aberrants_clients_inf)}")
print(f"Nombre total de clients : {len(data_clients)}")

# Détection stricte (> 8 écarts-types)
aberrants_clients_sup_v2 = data_clients[data_clients['Prix total'] > (moyenne_ca_client + 8 * std_ca_client)]
aberrants_clients_inf_v2 = data_clients[data_clients['Prix total'] < (moyenne_ca_client - 8 * std_ca_client)]

print(f"Nombre de clients avec CA aberrant (> 8 écarts-types) : {len(aberrants_clients_sup_v2)}")
print(f"Nombre de clients avec CA aberrant (< 8 écarts-types) : {len(aberrants_clients_inf_v2)}")

print(aberrants_clients_sup_v2[['client', 'Prix total']])
print(aberrants_clients_inf_v2[['client', 'Prix total']])

print("##########################################################################################")
print("##########################################################################################")
print("##########################################################################################")



####################################### Partie Graph  #########################################

plt.scatter(df_v2['unit_price'], df_v2['quantity'])
plt.xlabel('Prix Unitaire')
plt.ylabel('Quantité Vendue')
plt.title("Relation entre Prix et Quantité à partir de la data pur")
plt.show()



# 1️⃣ Histogramme : Distribution du chiffre d'affaires total par commande

# Distribution du CA par Pizza
plt.figure(figsize=(10,5))
plt.hist(data_pizzas['Chiffre Affaire'], bins=30, color='orange', edgecolor='black')
plt.title("Distribution du Chiffre d'Affaires par Pizza")
plt.xlabel("Chiffre d'Affaires (€)")
plt.ylabel("Nombre de Pizzas Vendues")
plt.grid(True)
plt.show()

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

# Pizzas : Prix unitaire vs Quantité vendue
plt.figure(figsize=(8,5))
plt.scatter(data_pizzas['Prix unitaire'], data_pizzas['Quantité total commande'], color='green')
plt.xlabel('Prix Unitaire (€)')
plt.ylabel('Quantité Totale Commandée')
plt.title("Relation Prix Unitaire / Quantité (Pizzas)")
plt.grid(True)
plt.show()

# Clients : Total dépensé vs Quantité de pizza
plt.figure(figsize=(8,5))
plt.scatter(data_clients['Prix total'], data_clients['Quantite de pizza'], color='purple')
plt.xlabel('Total Dépensé (€)')
plt.ylabel('Quantité Totale Commandée')
plt.title("Relation Dépenses / Quantité (Clients)")
plt.grid(True)
plt.show()






############################################################

order_id = 18845.00
pizzas = df[df['order_id'] == order_id]['pizza_name'].values
quantities = df[df['order_id'] == order_id]['quantity'].values

print(f"Pizzas commandées pour la commande {order_id} : {pizzas}")
print(f"Quantités commandées : {quantities}")























