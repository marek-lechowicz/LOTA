import os
import torch
import datetime
import numpy as np
import pandas as pd

# Import modules with alternative names
from util import set_random_seed as seed_generator
from util import poly_lr as learning_rate_adjuster
from loader import get_val_loader as acquire_validation_dataset
from config import ConfigurationManager as Configurator
from model import model as DeepLearningModel

# Configure image loading behavior
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_validation_settings():
    """Create specialized configuration for model assessment"""
    settings = Configurator().parse()
    settings.isTrain = False
    settings.isVal = True
    settings.isTest = True
    return settings


def compute_metrics(y_true, y_scores):
    """Calculate comprehensive evaluation metrics using numpy"""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int)

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Precision, Recall
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # MCC
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0

    # AUC ROC
    try:
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        
        if tps[-1] == 0 or fps[-1] == 0:
            auc = 0.5 # Default for single-class
        else:
            tpr = tps / tps[-1]
            fpr = fps / fps[-1]
            auc = np.trapz(tpr, fpr)
    except:
        auc = 0.0

    # Average Precision (AP)
    try:
        if np.sum(y_true) == 0:
            ap = 0.0
        else:
            # Sort by scores
            indices = np.argsort(y_scores)[::-1]
            y_true_sorted = y_true[indices]
            
            tps = np.cumsum(y_true_sorted)
            fps = np.cumsum(1 - y_true_sorted)
            
            precisions = tps / (tps + fps)
            recalls = tps / np.sum(y_true)
            
            # Prepend/Append for calculation
            precisions = np.concatenate(([1], precisions))
            recalls = np.concatenate(([0], recalls))
            
            ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    except:
        ap = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "auc": auc,
        "ap": ap
    }


def assess_model_performance(
        validation_datasets,
        neural_network,
        results_directory
):
    """Evaluate neural network performance across validation datasets"""
    neural_network.eval()
    aggregate_correct = aggregate_samples = 0

    with torch.no_grad():
        for dataset in validation_datasets:
            dataset_labels = []
            dataset_scores = []
            dataset_paths = []

            dataset_identifier = dataset['name']
            ai_data_loader = dataset['val_ai_loader']
            natural_data_loader = dataset['val_nature_loader']

            print(f"[Evaluating dataset: {dataset_identifier}]")

            # Process both loaders
            for loader_name, data_loader in [("AI", ai_data_loader), ("Natural", natural_data_loader)]:
                for image_batch, target_labels, paths in data_loader:
                    image_batch = image_batch.cuda()
                    
                    predictions = neural_network(image_batch)
                    prediction_scores = torch.sigmoid(predictions).flatten()

                    dataset_labels.extend(target_labels.cpu().numpy().tolist())
                    dataset_scores.extend(prediction_scores.cpu().numpy().tolist())
                    dataset_paths.extend(paths)

            # Compute metrics for this dataset
            m = compute_metrics(dataset_labels, dataset_scores)
            
            print(f"--- Results for {dataset_identifier} ---")
            print(f"Accuracy:  {m['accuracy']:.4f}")
            print(f"AUC ROC:   {m['auc']:.4f}")
            print(f"Avg Prec:  {m['ap']:.4f}")
            print(f"Precision: {m['precision']:.4f}")
            print(f"Recall:    {m['recall']:.4f}")
            print(f"MCC:       {m['mcc']:.4f}")

            # Save detailed results to CSV
            df = pd.DataFrame({
                'path': dataset_paths,
                'label': dataset_labels,
                'score': dataset_scores
            })
            csv_name = f"detailed_results_{dataset_identifier}.csv"
            csv_path = os.path.join(results_directory, csv_name)
            df.to_csv(csv_path, index=False)
            print(f"Detailed results saved to: {csv_path}")

            # Global aggregation
            aggregate_correct += np.sum(np.array(dataset_labels) == (np.array(dataset_scores) > 0.5))
            aggregate_samples += len(dataset_labels)

            # Compatibility with old parser in cross_experiment_pipeline.py
            # The parser looks for "Subset Performance: 0.1234"
            print(f"Subset Performance: {m['accuracy']:.4f}")

    # Compute overall performance
    if aggregate_samples > 0:
        overall_performance = aggregate_correct / aggregate_samples
        print(f"[Global Accuracy: {overall_performance:.4f}]")


def configure_computation_device(device_id):
    """Set computational hardware environment"""
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"Selected computation device: GPU {device_id}")


def execute_evaluation_procedure():
    """Main evaluation workflow execution"""
    # Initialize random number generation
    seed_generator()

    # Load configuration settings
    primary_config = Configurator().parse()
    validation_config = generate_validation_settings()

    # Prepare validation data
    print('Preparing validation datasets...')
    validation_datasets = acquire_validation_dataset(validation_config)

    # Configure hardware environment
    configure_computation_device(primary_config.gpu_id)

    # Initialize neural architecture
    network_instance = DeepLearningModel().cuda()

    # Load pre-trained parameters if specified
    if primary_config.load is not None:
        network_instance.load_state_dict(torch.load(primary_config.load))
        print(f'Loaded model parameters from {primary_config.load}')

    # Create results storage location
    results_path = primary_config.save_path
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    print("Commencing model evaluation")
    assess_model_performance(validation_datasets, network_instance, results_path)


if __name__ == '__main__':
    execute_evaluation_procedure()
