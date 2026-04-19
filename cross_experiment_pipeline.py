import os
import subprocess
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Nazwy modeli zgodne z loader.py
MODEL_NAMES = [
    'flux_1_dev',
    'flux_fill_flux_1_dev',
    'flux_fill_real_rescaled',
    'flux_fill_sd_3_5_large',
    'sd_1_5',
    'sd_3_5_large',
    'sdxl_turbo',
    'z_image_turbo'
]

PYTHON_EXEC = "./.venv/bin/python"

def run_command(cmd):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def train_all_models(epochs=10):
    for i in range(8):
        choices = ["0"] * 8
        choices[i] = "1"
        
        save_path = f"results/train_on_{MODEL_NAMES[i]}"
        print(f"\n>>> TRENING: {MODEL_NAMES[i]}...")
        
        train_cmd = [
            PYTHON_EXEC, "train.py",
            "--choices", *choices,
            "--save_path", save_path,
            "--epoch", str(epochs)
        ]
        run_command(train_cmd)

def evaluate_and_build_matrix():
    matrix = np.zeros((8, 8))
    
    for i in range(8):
        model_path = f"results/train_on_{MODEL_NAMES[i]}/Network_best.pth"
        if not os.path.exists(model_path):
            print(f"Pominięto: Brak modelu {model_path}")
            continue
            
        print(f"\n>>> EWALUACJA modelu wytrenowanego na {MODEL_NAMES[i]}...")
        # Testujemy na wszystkich 8 zbiorach (choices 1 1 1...)
        choices = ["1"] * 8
        test_cmd = [
            PYTHON_EXEC, "test.py",
            "--choices", *choices,
            "--load", model_path
        ]
        
        output = run_command(test_cmd)
        
        # Parsowanie wyników accuracy z test.py
        datasets_found = re.findall(r"\[Evaluating dataset: (.*?)\]", output)
        performances = re.findall(r"Subset Performance: (0\.\d+)", output)
        
        perf_map = dict(zip(datasets_found, [float(p) for p in performances]))
        
        for j, test_name in enumerate(MODEL_NAMES):
            if test_name in perf_map:
                matrix[i, j] = perf_map[test_name]
                
    return matrix

def plot_matrix(matrix):
    df = pd.DataFrame(matrix, index=MODEL_NAMES, columns=MODEL_NAMES)
    
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")
    sns.heatmap(
        df, annot=True, fmt=".4f", cmap="YlGnBu", 
        cbar_kws={'label': 'Accuracy'}, linewidths=0.5
    )
    
    plt.title("Cross-Model Evaluation (Accuracy Matrix)", fontsize=16, pad=20)
    plt.xlabel("Test Set (AI Model)", fontsize=12)
    plt.ylabel("Training Set (Trained Model)", fontsize=12)
    plt.tight_layout()
    
    output_path = "results/accuracy_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nWykres zapisano w: {output_path}")

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # 1. Trenuj każdy model z osobna
    train_all_models(epochs=10) # możesz zwiększyć liczbę epok
    
    # 2. Ewaluacja krzyżowa
    matrix = evaluate_and_build_matrix()
    
    # 3. Generowanie macierzy (confusion matrix style)
    plot_matrix(matrix)
