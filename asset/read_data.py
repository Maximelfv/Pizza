import pandas as pd
import numpy as np


df = pd.read_csv("./dataset/pizza_sales.csv")


# ============= Création des dataFrame pour les pizzas et pour les clients ===================

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

data_pizzas['Horodatage'] = pd.to_datetime(data_pizzas['Horodatage'], dayfirst=True)
data_pizzas['Chiffre Affaire'] = data_pizzas['Prix unitaire']*data_pizzas['Quantité total commande']
# Le data_pizza regroupe toute les differentes pizzas possible de commander (nom + taille)
# Et informe sur le nombre de fois elle a était commandé et son prix unitaire et de son CA et la date de commande



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

data_clients['Horodatage'] = pd.to_datetime(data_clients['Horodatage'], dayfirst=True)
# Le data_clients regroupe toute les commandes de pizza par client 
# Et informe sur le nombre de pizzas commandé, le prix total de la commande et la date de commande par client



#############################################################################################
#############################################################################################
#############################################################################################

def info(data):
    moyenne_ca = np.mean(data)
    mediane_ca = np.median(data)
    max_ca = np.max(data)
    min_ca = np.min(data)
    
    return moyenne_ca,mediane_ca,max_ca,min_ca


####################################################################################
####################################################################################
####################################################################################


############################### PARTIE IA ##########################################
# ================================= Mois ==========================================
Mois = []
Quantite_pizza_mois = []
Chiffre_Affaire_mois = []

df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)

for i in range(1, 13):
    Mois.append(i)
    Quantite_pizza_mois.append(df[df['order_date'].dt.month == i]['quantity'].sum()) 
    Chiffre_Affaire_mois.append(df[df['order_date'].dt.month == i]['total_price'].sum())

data_pizzas_mois = pd.DataFrame({
    'Mois' : Mois,
    'Quantité total commande' : Quantite_pizza_mois,
    'Chiffre Affaire' : Chiffre_Affaire_mois,
    })  

# Le data_pizza_mois regroupe toute les commandes de pizza par mois
# Et informe sur le nombre de pizzas commandé, le prix total de la commande par mois



# ================================= Jour ==========================================
df_journalier = df.groupby("order_date").agg({
    "quantity": "sum",
    "total_price": "sum"
}).reset_index()

# ================================= Semaine ========================================
df_semaine = df.groupby(pd.Grouper(key="order_date", freq="W")).agg({
    "quantity": "sum",
    "total_price": "sum"
}).reset_index()





def __main__():
    # Affichage des données
    print("DataFrame générale :")
    print(df.head())
    print("DataFrame des pizzas :")
    print(data_pizzas.head())
    print("\nDataFrame des clients :")
    print(data_clients.head())
    print("\nDataFrame des ventes par mois :")
    print(data_pizzas_mois.head())

if __name__ == "__main__":
    __main__()
