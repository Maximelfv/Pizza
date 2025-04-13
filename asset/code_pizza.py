import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv("./dataset/pizza_sales.csv")


df_journalier = df.groupby("order_date").agg({
    "quantity": "sum",
    "total_price": "sum"
}).reset_index()




X = df_journalier[["total_price"]]
y = df_journalier["quantity"] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

modele= LinearRegression()
modele.fit(X_train, y_train)


y_pred = modele.predict(X_test)
score_precision = r2_score(y_test, y_pred)
erreur_quadratique = mean_squared_error(y_test, y_pred)

print("Score de précision :", round(score_precision,3))
print("Erreur quadratique moyenne :", round(erreur_quadratique,5))



#Exemple de prédiction

prediction=modele.predict(pd.DataFrame([[36]], columns=["total_price"]))

print("\n Nombre de pizzas prédit :",round(prediction[0]))




modele.fit(X, y)
df_journalier["prediction"] = modele.predict(X)
df_journalier["prediction"] = df_journalier["prediction"].round().astype(int)
df_journalier["quantity"] = df_journalier["quantity"].round().astype(int)

print("\n Comparatif réel vs prédit (par jour) :")
print(df_journalier[["order_date", "quantity", "prediction"]].sort_values("order_date").head(11))




plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Vraie quantité de pizzas vendues")
plt.ylabel("Quantité prédite")
plt.title("Quantité réelle vs prédite")
plt.grid(True)
plt.tight_layout()
plt.show()
