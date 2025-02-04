import shutil
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from datetime import datetime
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# creates a new training folder with unique name, ex. t1, t2 ...
def create_experiment_folder(base_folder="results"):
    experiment_id = len([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]) + 1
    experiment_folder = os.path.join(base_folder, f"t{experiment_id}")
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder

def save_plot(plt, path):
    plt.savefig(path, format='png', dpi=300)
    plt.close()


# convert float32 -> float
def convert_np_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj

df = pd.read_csv('../2_clean_data/results/otodom_houses_cleaned.csv', delimiter=';')


# prepare data
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)


# load model parameters from JSON
with open('model_params.json', 'r') as file:
    model_params = json.load(file)

models = {
    name: eval(model_params[name]['model'])(**model_params[name]['params'])
    for name in model_params
}


results = []
experiment_folder = create_experiment_folder()

# save scaler to current experiment folder
scaler_path = os.path.join(experiment_folder, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# make folder for models for current experiment
models_folder = os.path.join(experiment_folder, "models")
os.makedirs(models_folder, exist_ok=True)

# train all models from model_params.json file
for name, model in models.items():
    model.fit(X_train_scaled, y_train_log)
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)


    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        result = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        importances = result.importances_mean

    feature_importance_results = {
        "features": list(X_test.columns),
        "importances": list(importances)
    }
    
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name} - MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

    model_path = os.path.join(models_folder, f"{name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)
    print(f"Model {name} saved to {model_path}")

    feature_importance_path = os.path.join(models_folder, f"{name.replace(' ', '_')}_future_importances.json")
    with open(feature_importance_path, 'w') as f:
        json.dump(feature_importance_results, f, indent=4, default=convert_np_types)
    print(f"Future importances for {name} saved to {feature_importance_path}")



# save results of experiment to json
results_path = os.path.join(experiment_folder, "results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)


# plot best model prediction result
best_model_name = min(results, key=lambda x: x["RMSE"])['Model']
best_model = models[best_model_name]

best_results_folder = "best_results"
os.makedirs(best_results_folder, exist_ok=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=np.expm1(best_model.predict(X_test_scaled)), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted Price ({best_model_name})")
save_plot(plt, os.path.join(experiment_folder, "actual_vs_predicted.png"))
shutil.copy(
    os.path.join(experiment_folder, "actual_vs_predicted.png"),
    os.path.join(best_results_folder, "actual_vs_predicted.png")
)

# plot RMSE comparison
plt.figure(figsize=(10, 15))
sns.barplot(x="Model", y="RMSE", data=pd.DataFrame(results), palette="coolwarm")
plt.yscale("log")
plt.xticks(rotation=45, ha="right")
plt.title("Model Comparison - RMSE")
save_plot(plt, os.path.join(experiment_folder, "rmse_comparison.png"))
shutil.copy(
    os.path.join(experiment_folder, "rmse_comparison.png"),
    os.path.join(best_results_folder, "rmse_comparison.png")
)

# plot MAE comparison
plt.figure(figsize=(10, 15))
sns.barplot(x="Model", y="MAE", data=pd.DataFrame(results), palette="coolwarm")
plt.yscale("log")
plt.xticks(rotation=45, ha="right")
plt.title("Model Comparison - MAE")
save_plot(plt, os.path.join(experiment_folder, "mae_comparison.png"))
shutil.copy(
    os.path.join(experiment_folder, "mae_comparison.png"),
    os.path.join(best_results_folder, "mae_comparison.png")
)


# save best model results
best_results_folder = "best_results"
os.makedirs(best_results_folder, exist_ok=True)

best_model_path = os.path.join(best_results_folder, f"{best_model_name.replace(' ', '_')}.pkl")
joblib.dump(best_model, best_model_path)
print(f"Best model {best_model_name} saved to {best_model_path}")

scaler_path = os.path.join(best_results_folder, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

best_results_path = os.path.join(best_results_folder, "results.json")
with open(best_results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nBest model: {best_model_name}")