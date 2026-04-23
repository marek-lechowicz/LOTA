import os
import subprocess
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

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
    results_summary = []

    for i, train_exp in enumerate(EXPERIMENTS):
        train_name = train_exp['name']
        model_path = f"results/train_on_{train_name}/Network_best.pth"
        if not os.path.exists(model_path):
            print(f"Pominięto: Brak modelu {model_path}")
            continue
            
        print(f"\n>>> EWALUACJA modelu wytrenowanego na {train_name}...")
        
        # Folder na wyniki ewaluacji tego konkretnego modelu
        eval_save_dir = f"results/train_on_{train_name}/evaluation_results"
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)

        for j, test_exp in enumerate(EXPERIMENTS):
            test_name = test_exp['name']
            choices = ["0"] * 8
            choices[MODEL_TO_CHOICE_IDX[test_exp["fake"]]] = "1"
            
            test_cmd = [
                PYTHON_EXEC, "test.py",
                "--choices", *choices,
                "--real_source", test_exp["real"],
                "--load", model_path,
                "--save_path", eval_save_dir
            ]
            
            output = run_command(test_cmd)
            
            # Wyciąganie metryk
            # Szukamy bloku: --- Results for ... ---
            m_acc = re.search(r"Accuracy:\s+(0\.\d+)", output)
            m_auc = re.search(r"AUC ROC:\s+(0\.\d+)", output)
            m_ap = re.search(r"Avg Prec:\s+(0\.\d+)", output)
            m_prec = re.search(r"Precision:\s+(0\.\d+)", output)
            m_rec = re.search(r"Recall:\s+(0\.\d+)", output)
            m_mcc = re.search(r"MCC:\s+(-?0\.\d+|0\.0)", output)
            
            metrics = {
                "train_on": train_name,
                "test_on": test_name,
                "accuracy": float(m_acc.group(1)) if m_acc else 0.0,
                "auc": float(m_auc.group(1)) if m_auc else 0.0,
                "ap": float(m_ap.group(1)) if m_ap else 0.0,
                "precision": float(m_prec.group(1)) if m_prec else 0.0,
                "recall": float(m_rec.group(1)) if m_rec else 0.0,
                "mcc": float(m_mcc.group(1)) if m_mcc else 0.0,
            }
            results_summary.append(metrics)
            
            # Wpisanie do macierzy (dla wstecznej kompatybilności wykresu)
            matrix[i, j] = metrics["accuracy"]
                
    # Zapisanie zbiorczego raportu
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("results/cross_evaluation_detailed_metrics.csv", index=False)
    print(f"\nZbiorczy raport metryk zapisano w: results/cross_evaluation_detailed_metrics.csv")

    return summary_df, exp_names

def plot_matrix(df_summary, exp_names, metric_name, title_label):
    n = len(exp_names)
    matrix = np.zeros((n, n))
    
    # Budujemy macierz z dataframe'u
    for _, row in df_summary.iterrows():
        try:
            i = exp_names.index(row['train_on'])
            j = exp_names.index(row['test_on'])
            matrix[i, j] = row[metric_name]
        except ValueError:
            continue

    df = pd.DataFrame(matrix, index=exp_names, columns=exp_names)
    
    plt.figure(figsize=(16, 14))
    sns.set_theme(style="white")
    
    # Dostosowanie skali kolorów (MCC może być ujemne)
    cmap = "YlGnBu"
    if metric_name == "mcc":
        cmap = "RdBu_r"
        vmin, vmax = -1, 1
    else:
        vmin, vmax = 0, 1

    sns.heatmap(
        df, annot=True, fmt=".4f", cmap=cmap, vmin=vmin, vmax=vmax,
        cbar_kws={'label': title_label}, linewidths=0.5
    )
    
    plt.title(f"Cross-Model Evaluation ({title_label} Matrix)", fontsize=16, pad=20)
    plt.xlabel("Test Set (AI Model with specific Real Source)", fontsize=12)
    plt.ylabel("Training Set (Trained Model)", fontsize=12)
    plt.tight_layout()
    
    output_path = f"results/matrix_{metric_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Wykres {metric_name} zapisano w: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Experiment Pipeline Control")
    parser.add_argument("--plot_only", default=True, action="store_true", help="Skip training/eval and only generate plots from results/cross_evaluation_detailed_metrics.csv")
    args = parser.parse_args()

    if not os.path.exists("results"):
        os.makedirs("results")
        
    exp_names = [e["name"] for e in EXPERIMENTS]
    
    if args.plot_only:
        csv_path = "results/cross_evaluation_detailed_metrics.csv"
        if os.path.exists(csv_path):
            print(f">>> TRYB PLOT_ONLY: Wczytywanie wyników z {csv_path}")
            summary_df = pd.read_csv(csv_path)
        else:
            print(f"BŁĄD: Nie znaleziono pliku {csv_path}. Uruchom bez --plot_only po raz pierwszy.")
            exit(1)
    else:
        train_all_models(epochs=10) 
        summary_df, _ = evaluate_and_build_matrix()
    
    METRICS_TO_PLOT = [
        ("accuracy", "Accuracy"),
        ("auc", "AUC ROC"),
        ("ap", "Average Precision"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("mcc", "Matthews Correlation Coefficient")
    ]
    
    print("\n>>> GENEROWANIE WYKRESÓW...")
    for metric_id, metric_label in METRICS_TO_PLOT:
        plot_matrix(summary_df, exp_names, metric_id, metric_label)
