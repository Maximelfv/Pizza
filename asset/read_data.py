import pandas as pd
import numpy as np




df = pd.read_csv("./dataset/pizza_sales.csv")
print(df.head())
print(df.info())


# Le chiffre d'affaire existe deja (total_price) je vais juste le remplacer

#  Ajout des chiffres d'affaires
df_v2 = df.drop(columns='total_price')
df_v2['Chiffre_Affaires'] = df['unit_price'] * df['quantity']

print(df_v2.info())
print(df_v2.head())



# Création des dataFrame pour les pizzas et pour les clients

########################### PIZZA  ###########################################################

pizza_unique_name = []
Quantite = []
Prix_unitaire = []
horodatage_pizza = []

for pizza in df['pizza_name_id'].unique():
    pizza_unique_name.append(pizza)
    Quantite.append(df[df['pizza_name_id']==pizza]['quantity'].sum())
    Prix_unitaire.append(df[df['pizza_name_id']==pizza]['unit_price'].values[0])
    horodatage_pizza.append(df[df['pizza_name_id']==pizza]['order_date'].values[0])

data_pizzas = pd.DataFrame({
    'Horodatage' : horodatage_pizza,
    'Name' : pizza_unique_name,
    'Quantité total commande' : Quantite,
    'Prix unitaire' : Prix_unitaire,
    })

data_pizzas['Horodatage'] = pd.to_datetime(data_pizzas['Horodatage'])

data_pizzas['Chiffre Affaire'] = data_pizzas['Prix unitaire']*data_pizzas['Quantité total commande']

print(data_pizzas)
# Le data_pizza regroupe toute les differentes pizzas possible de commander (nom + taille)
# Et informe sur le nombre de fois elle a était commandé et son prix unitaire et de son CA

########################### CLIENT   ########################################################

commande_client = []
Quantite_pizza_client = []
prix_total_client = []
horodatage_client = []

for client in df['order_id'].unique():
    commande_client.append(client)
    Quantite_pizza_client.append(df[df['order_id']==client]['quantity'].sum())
    prix_total_client.append(df[df['order_id']==client]['total_price'].sum())
    horodatage_client.append(df[df['order_id']==client]['order_date'].values[0])

data_clients = pd.DataFrame({
    'Horodatage' : horodatage_client,
    'client' : commande_client,
    'Quantite de pizza' : Quantite_pizza_client,
    'Prix total' : prix_total_client,
    })

data_clients['Horodatage'] = pd.to_datetime(data_clients['Horodatage'])

print(data_clients)


#############################################################################################
#############################################################################################
#############################################################################################

def info(data):
    moyenne_ca = np.mean(data)
    mediane_ca = np.median(data)
    max_ca = np.max(data)
    min_ca = np.min(data)
    
    print(f" Moyenne CA: {moyenne_ca}\n Médiane CA: {mediane_ca},\n Max CA: {max_ca},\n Min CA: {min_ca}\n")
    return moyenne_ca,mediane_ca,max_ca,min_ca


####################################################################################
####################################################################################
####################################################################################


############################### PARTIE IA ##########################################
from datetime import datetime

Mois = []
Quantite_pizza_mois = []
Chiffre_Affaire_mois = []

df['order_date'] = pd.to_datetime(df['order_date'])

for i in range(1, 13):
    Mois.append(i)
    Quantite_pizza_mois.append(df[df['order_date'].dt.month == i]['quantity'].sum()) 
    Chiffre_Affaire_mois.append(df[df['order_date'].dt.month == i]['total_price'].sum())

data_pizzas_mois = pd.DataFrame({
    'Mois' : Mois,
    'Quantité total commande' : Quantite_pizza_mois,
    'Chiffre Affaire' : Chiffre_Affaire_mois,
    })  

print(data_pizzas_mois)
