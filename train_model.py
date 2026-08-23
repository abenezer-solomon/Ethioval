import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load local Zenodo dataset
print("Loading Addis Ababa real estate dataset...")
df = pd.read_csv("real-estate_dataset_addis-ababa_v1.csv")

# 2. Filter target leakage columns
leakage_cols = [
    'price_sqm', 'price_adj', 'price_adj_sqm', 
    'size_sqm_is_imputed', 'id', 'url', 'address_alt', 'seller_address'
]
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

# 3. Target definition (Drop rows without price)
target = 'price'
df = df.dropna(subset=[target])

# 4. Feature Selection
num_cols = ['size_sqm', 'num_bedrooms', 'num_bathrooms', 'dist_meskel_square', 'lat', 'lng']
cat_cols = ['subcity', 'property_type', 'listing_type', 'furnishing']

# Keep columns present in the dataset
num_cols = [c for c in num_cols if c in df.columns]
cat_cols = [c for c in cat_cols if c in df.columns]

X = df[num_cols + cat_cols]
y = df[target]

# 5. Preprocessing Pipeline
num_transformer = SimpleImputer(strategy='median')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# 6. Train/Test Split & Fit
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_trans = preprocessor.fit_transform(X_train)
X_valid_trans = preprocessor.transform(X_valid)

model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
model.fit(X_train_trans, y_train)

# 7. Evaluate & Save Pipeline Artifacts
preds = model.predict(X_valid_trans)
print(f"Validation MAE: {mean_absolute_error(y_valid, preds):,.2f} ETB")

joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(model, 'xgboost_model.joblib')
print("Model artifacts successfully saved as joblib files!")