import json
import os

def get_available_models(results_dir = '3_train/results'):
    available_models = {}

    for folder in os.listdir(results_dir):
        models_dir = os.path.join(results_dir, folder, 'models')
        results_file = os.path.join(results_dir, folder, 'results.json')

        if os.path.isdir(models_dir) and os.path.isfile(results_file):
            for file in os.listdir(models_dir):
                if file.endswith('.pkl'):
                    model_name = file.replace('.pkl', '')
                    available_models[f"{folder}/{model_name}"]={
                        "model_path": os.path.join(models_dir, file),
                        "scaler_path": os.path.join(results_dir, folder, 'scaler.pkl'),
                        "results_path": results_file
                    }

    return available_models


def get_trainings(results_dir = '3_train/results'):
    return os.listdir(results_dir)

def get_models_from_trainig(training_path):
    available_models = {}
    models_dir = os.path.join(training_path, 'models')
    scaler_path = os.path.join(training_path, 'scaler.pkl')
    results_file = os.path.join(training_path, 'results.json')

    for file in os.listdir(models_dir):
        if file.endswith('.pkl'):
            model_name = file.replace('.pkl', '')
            available_models[model_name]={
                "model_path": os.path.join(models_dir, file),
                "scaler_path": scaler_path,
                "results_path": results_file
            }

    return available_models


def get_mae_from_results(results_path, model_name):
    with open(results_path, 'r') as file:
        results = json.load(file)

    for result in results:
        if result["Model"] == model_name:
            return result["MAE"]
    return None