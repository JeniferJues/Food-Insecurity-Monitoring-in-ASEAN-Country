import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("dataset/predict_df.csv")

# ==========================
# SELECTED FEATURES
# ==========================
selected_features = [
    'Average value of food production (constant 2004-2006 I$/cap) (3-year average)',
    'Cereal import dependency ratio (percent) (3-year average)',
    'Incidence of caloric losses at retail distribution level (percent)',
    'Per capita food production variability (constant 2004-2006 thousand int$ per capita)',
    'Per capita food supply variability (kcal/cap/day)',
    'Percent of arable land equipped for irrigation (percent) (3-year average)',
    'Percentage of children under 5 years of age who are overweight (modelled estimates) (percent)',
    'water access',
    'Percentage of population using at least basic sanitation services (percent)',
    'irrigation',
    'Political stability and absence of violence/terrorism (index)',
    'Prevalence of anemia among women of reproductive age (15-49 years)',
    'Share of dietary energy supply derived from cereals, roots and tubers (kcal/cap/day) (3-year average)',
    'Value of food imports in total merchandise exports (percent) (3-year average)',
    'Consumer Prices, General Indices (2015 = 100)'
]

X = df[selected_features]
y = df["Food Insecurity Rate"]

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# EVALUATE
# ==========================
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("MAE:", round(mae, 3))
print("R2:", round(r2, 3))

# ==========================
# SAVE MODEL + FEATURES
# ==========================
joblib.dump(model, "models/prediction/food_model.pkl")
joblib.dump(selected_features, "models/prediction/feature_columns.pkl")

print("Prediction model saved successfully.")
