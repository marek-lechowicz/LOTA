import os
import subprocess
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Słownik indeksów z loader.py (argument --choices)
MODEL_TO_CHOICE_IDX = {
    'flux_1_dev': 0,
    'flux_fill_flux_1_dev': 1,
    'flux_fill_real_rescaled': 2,
    'flux_fill_sd_3_5_large': 3,
    'sd_1_5': 4,
    'sd_3_5_large': 5,
    'sdxl_turbo': 6,
    'z_image_turbo': 7
}

# 11 eksperymentów z data.py z ResNeta
EXPERIMENTS = [
    # General models vs primary_real_source_name ("real")
    {"name": "flux_1_dev__vs__real", "fake": "flux_1_dev", "real": "real"},
    {"name": "flux_fill_flux_1_dev__vs__real", "fake": "flux_fill_flux_1_dev", "real": "real"},
    {"name": "flux_fill_real_rescaled__vs__real", "fake": "flux_fill_real_rescaled", "real": "real"},
    {"name": "flux_fill_sd_3_5_large__vs__real", "fake": "flux_fill_sd_3_5_large", "real": "real"},
    {"name": "sd_1_5__vs__real", "fake": "sd_1_5", "real": "real"},
    {"name": "sd_3_5_large__vs__real", "fake": "sd_3_5_large", "real": "real"},
    {"name": "sdxl_turbo__vs__real", "fake": "sdxl_turbo", "real": "real"},
    {"name": "z_image_turbo__vs__real", "fake": "z_image_turbo", "real": "real"},
    
    # Inpainting model vs its exact rescaled/generated source
    {"name": "flux_fill_real_rescaled__vs__real_rescaled", "fake": "flux_fill_real_rescaled", "real": "real_rescaled"},
    {"name": "flux_fill_flux_1_dev__vs__flux_1_dev", "fake": "flux_fill_flux_1_dev", "real": "flux_1_dev"},
    {"name": "flux_fill_sd_3_5_large__vs__sd_3_5_large", "fake": "flux_fill_sd_3_5_large", "real": "sd_3_5_large"},
]

PYTHON_EXEC = "./.venv/bin/python"

def run_command(cmd, capture=True):
    print(f"Executing: {' '.join(cmd)}")
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.stdout
    else:
        result = subprocess.run(cmd)
        return result.returncode == 0

def train_all_models(epochs=10):
    for exp in EXPERIMENTS:
        save_path = f"results/train_on_{exp['name']}"
        finished_flag = os.path.join(save_path, "training_finished.flag")
        
        if os.path.exists(finished_flag):
            print(f"\n>>> POMINIĘTO TRENING: {exp['name']} (znaleziono flagę ukończenia)")
            continue
            
        choices = ["0"] * 8
        choices[MODEL_TO_CHOICE_IDX[exp["fake"]]] = "1"
        
        print(f"\n>>> TRENING: {exp['name']}...")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        train_cmd = [
            PYTHON_EXEC, "train.py",
            "--choices", *choices,
            "--real_source", exp["real"],
            "--save_path", save_path,
            "--epoch", str(epochs)
        ]
        
        success = run_command(train_cmd, capture=False)
        if success:
            with open(finished_flag, "w") as f:
                f.write("Done")
        else:
            print(f"\n[!] Trening przerwany przy eksperymencie {exp['name']}. Skrypt cross_experiment został zatrzymany.")
            break


def evaluate_and_build_matrix():
    n = len(EXPERIMENTS)
    matrix = np.zeros((n, n))
    exp_names = [e["name"] for e in EXPERIMENTS]
    
    for i, train_exp in enumerate(EXPERIMENTS):
        model_path = f"results/train_on_{train_exp['name']}/Network_best.pth"
        if not os.path.exists(model_path):
            print(f"Pominięto: Brak modelu {model_path}")
            continue
            
        print(f"\n>>> EWALUACJA modelu wytrenowanego na {train_exp['name']}...")
        
        for j, test_exp in enumerate(EXPERIMENTS):
            choices = ["0"] * 8
            choices[MODEL_TO_CHOICE_IDX[test_exp["fake"]]] = "1"
            
            test_cmd = [
                PYTHON_EXEC, "test.py",
                "--choices", *choices,
                "--real_source", test_exp["real"],
                "--load", model_path
            ]
            
            output = run_command(test_cmd)
            
            datasets_found = re.findall(r"\[Evaluating dataset: (.*?)\]", output)
            performances = re.findall(r"Subset Performance: (0\.\d+)", output)
            
            perf_map = dict(zip(datasets_found, [float(p) for p in performances]))
            
            # Nasze test.py ewaluowało choices wskazujące celowy dataset,
            # odczytujemy więc jego wynik bezpośrednio jeśli log się zgadza z kluczem "fake".
            if test_exp["fake"] in perf_map:
                matrix[i, j] = perf_map[test_exp["fake"]]
                
    return matrix, exp_names

def plot_matrix(matrix, exp_names):
    df = pd.DataFrame(matrix, index=exp_names, columns=exp_names)
    
    plt.figure(figsize=(16, 14))
    sns.set_theme(style="white")
    sns.heatmap(
        df, annot=True, fmt=".4f", cmap="YlGnBu", 
        cbar_kws={'label': 'Accuracy'}, linewidths=0.5
    )
    
    plt.title("Cross-Model Evaluation (Accuracy Matrix)", fontsize=16, pad=20)
    plt.xlabel("Test Set (AI Model with specific Real Source)", fontsize=12)
    plt.ylabel("Training Set (Trained Model)", fontsize=12)
    plt.tight_layout()
    
    output_path = "results/accuracy_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nWykres zapisano w: {output_path}")

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
        
    train_all_models(epochs=10) 
    
    matrix, exp_names = evaluate_and_build_matrix()
    
    plot_matrix(matrix, exp_names)
