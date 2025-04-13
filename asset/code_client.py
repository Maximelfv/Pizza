import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from read_data import data_clients,info,df

#  Calcul moyenne, médianne, max et min du chiffre d'affaires des clients
moyenne_ca_client,mediane_ca_client,max_ca_client,min_ca_client = info(data_clients['Prix total'])

##  Affichage des clients dont chiffres d'affaires > 3*moyenne
clients_plus_rentables = data_clients[data_clients['Prix total'] > 3*moyenne_ca_client]
nb_clients_rentable = len(clients_plus_rentables)
print(f" Les clients les plus rentables :\n {clients_plus_rentables} \n Soit {nb_clients_rentable} clients dont le CA est supérieur à {3*moyenne_ca_client:.2f}")

# Calcul de la correlation entre le prix total et la quantité de pizza commandée
corr_client = data_clients['Prix total'].corr(data_clients['Quantite de pizza'])
print(f"Corrélation entre total dépensé et quantité commandée (Clients) : {corr_client:.2f}")

# Recherche du client le plus depensier
client_plus_depensier = data_clients[data_clients['Prix total'] == max(data_clients['Prix total'])]['client']
print(f"Voici le client qui a dépensé le plus : {client_plus_depensier.values[0]} avec {max(data_clients['Prix total'])} €")


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


####################################################################################
####################################################################################
####################################################################################


############################### PARTIE IA ##########################################



