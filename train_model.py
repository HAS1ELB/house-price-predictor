import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

np.random.seed(42)

n_samples = 1000

square_feet = np.random.uniform(500, 5000, n_samples)
bedrooms = np.random.randint(1, 7, n_samples)
bathrooms = np.random.randint(1, 5, n_samples)
age_years = np.random.uniform(0, 50, n_samples)
lot_size = np.random.uniform(1000, 20000, n_samples)
garage_spaces = np.random.randint(0, 4, n_samples)
neighborhood_score = np.random.uniform(1, 10, n_samples)

prix_base = 100000
price = (
    prix_base +
    (square_feet * 150) +
    (bedrooms * 20000) +
    (bathrooms * 15000) -
    (age_years * 2000) +
    (lot_size * 5) +
    (garage_spaces * 10000) +
    (neighborhood_score * 5000) +
    np.random.normal(0, 20000, n_samples)
)

df = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age_years': age_years,
    'lot_size': lot_size,
    'garage_spaces': garage_spaces,
    'neighborhood_score': neighborhood_score,
    'price': price
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/house_data.csv', index=False)

feature_names = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 
                 'lot_size', 'garage_spaces', 'neighborhood_score']
X = df[feature_names]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\nModèle, scaler et feature_names sauvegardés dans models/")
