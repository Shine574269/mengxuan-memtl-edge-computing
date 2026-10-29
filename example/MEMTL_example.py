"""
MEMTL example:
Read the configuration file and repeat experiments to compare the inference performance of
MTFNN and MEMTL with different number of prediction heads under different number of mobile terminals.

This enhanced version includes:
- Standardized logging for better traceability (outputs to console and experiment.log).
- Optional command-line argument for config file, maintaining original execution method.
- Progress bars (tqdm) for long-running loops.
- Improved code structure, readability, comments, and error handling.
- Robust path handling using pathlib.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import yaml
import logging
import argparse
from pathlib import Path
import os
import sys
current_script = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Force UTF-8 encoding on Windows to prevent garbled tqdm progress bar characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Assuming custom modules are in a reachable path (e.g., in the same project)
from utils.dataset import split_dataset2subsets_train
from utils.loss_function import custom_loss
from utils.cost_calc import over_all_cost
from utils.model_op import store_model
from model.MEMTL import MEMTL
from model.MTL import MTL

# --- 1. Setup Centralized Logging ---
# Configure logger to output to both console and a file for persistent records.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"), # Log to a file
        logging.StreamHandler()                # Also log to the console
    ]
)

# --- 2. Define Constants for Clarity ---
DECISION_THRESHOLD = 0.1 # Threshold for converting continuous prediction to binary decision
# This '6' likely represents 6 features per mobile user (e.g., CPU, memory, etc.).
FEATURES_PER_MU = 6
# Training parameters
EPOCHS = 35
LEARNING_RATE = 0.001

def _evaluate_memtl_performance(memtl, X_test, Y_test, mu_num):
    """
    Helper function to evaluate MEMTL model performance and select the best head.
    This encapsulates the complex evaluation logic.

    :param memtl: The trained MEMTL model.
    :param X_test: Test features.
    :param Y_test: Test labels.
    :param mu_num: Number of mobile users (determines output shape).
    :return: A tuple containing (predictions, mse, accuracy, summary_string, pick_rates)
    """
    logging.info(f">>>>>>>> Evaluating MEMTL on {X_test.shape[0]} test samples <<<<<<<<<<")

    start_time = time.time()
    
    # Use tf.function with XLA (jit_compile) for high-performance inference
    # This bypasses the heavy tf.keras Model.predict loop overhead
    @tf.function(jit_compile=True)
    def fast_predict(inputs):
        return memtl(inputs, training=False)
        
    x_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    
    # Warmup run to trace the graph and compile XLA (we don't time this)
    _ = fast_predict(x_tensor[:1])
    
    # Actual timed prediction
    start_time = time.time()
    outputs_tensors = fast_predict(x_tensor)
    outputs = [t.numpy() for t in outputs_tensors]
    
    end_time = time.time()
    prediction_time_per_sample = (end_time - start_time) * 1000 / X_test.shape[0]
    logging.info(f"MEMTL model prediction cost {prediction_time_per_sample:.4f}ms per sample.")

    global_mse = []
    global_decision_correctness = []
    global_pick_indices = []
    global_predictions = []

    # Iterate through each sample in the test set
    for j in range(X_test.shape[0]):
        mses = []
        decisions_correct = []

        # For each sample, evaluate every prediction head
        for i in range(memtl.head_num):
            prediction = outputs[i][j]
            mses.append(K.sum(K.square(prediction - Y_test[j, -mu_num:])))

            pre_decision_binary = np.where(prediction < DECISION_THRESHOLD, 0, 1)
            pre_class_int = pre_decision_binary.dot(1 << np.arange(pre_decision_binary.shape[-1] - 1, -1, -1))
            ground_truth_class = Y_test[j, -(mu_num + 1)]
            decisions_correct.append(bool(pre_class_int == ground_truth_class))

        # --- Head Selection Strategy ---
        # 1. Find all heads that made the correct offloading decision.
        # 2. Among these, pick the one with the lowest MSE.
        # 3. If no head was correct, fall back to picking the one with the overall lowest MSE.
        correct_decision_indices = [k for k, is_correct in enumerate(decisions_correct) if is_correct]

        if correct_decision_indices:
            mses_of_correct_heads = [mses[k] for k in correct_decision_indices]
            picked_index = correct_decision_indices[np.argmin(mses_of_correct_heads)]
        else:
            picked_index = np.argmin(mses)

        global_pick_indices.append(picked_index)
        global_mse.append(mses[picked_index])
        global_decision_correctness.append(decisions_correct[picked_index])
        global_predictions.append(outputs[picked_index][j])

    total_mse = np.array(K.sum(global_mse))
    avg_mse_per_sample = total_mse / X_test.shape[0]
    accuracy = sum(global_decision_correctness) / len(global_decision_correctness)

    logging.info(f"MEMTL overall MSE per sample: {avg_mse_per_sample:.4f}")
    logging.info(f"MEMTL overall decision accuracy: {accuracy:.4f}")

    summary = (f"MEMTL prediction cost {prediction_time_per_sample:.4f}ms per sample, "
               f"{avg_mse_per_sample:.4f}/{accuracy:.4f}")

    pick_rates = {}
    for h in range(memtl.head_num):
        rate = np.sum(np.array(global_pick_indices) == h) / len(global_pick_indices)
        pick_rates[h] = rate
        logging.info(f"MEMTL head {h + 1} pick rate: {rate:.4f}")

    return np.array(global_predictions), avg_mse_per_sample, accuracy, summary, pick_rates


def train_test_store(dataset_path: Path, head_range: tuple, env_params: tuple, model_path: Path):
    """
    Main experiment pipeline for a single dataset.
    Conducts training, testing, and result storage for a range of head numbers.
    """
    mtl_trained = False
    mtl_model = None
    memtl_model = None

    test_results_log = []
    # Use tqdm for a progress bar over the head_num loop (ascii=True prevents garbled chars on Windows)
    for head_num in tqdm(range(head_range[0], head_range[1] + 1), desc=f"Heads on {dataset_path.stem}", ascii=True):
        Xs, Ys, X_train_all, Y_train_all, test_raw_features = split_dataset2subsets_train(str(dataset_path), subset_num=head_num + 1)
        mu_num = int(Xs[0].shape[1] / FEATURES_PER_MU)

        env_summary = (f"mu_num={mu_num}, head_num={head_num}, subset_size={Xs[0].shape[0]}, "
                       f"shared_train_size={X_train_all.shape[0]}\n")
        logging.info(f"\n----- Starting experiment for head_num={head_num} -----")
        logging.info(env_summary.strip())

        # Train benchmark MTL model (only once per dataset)
        if not mtl_trained:
            mtl_trained = True
            mtl_model = MTL(mu_num)
            mtl_model.compile(loss=custom_loss, optimizer='adam')
            logging.info("Training benchmark MTL model...")
            start_time = time.time()
            mtl_model.fit(X_train_all, Y_train_all[:, -mu_num:], epochs=EPOCHS, verbose=0)
            logging.info(f"MTL training finished in {time.time() - start_time:.2f} seconds.")

        # Initialize or update MEMTL model
        if head_num == head_range[0]: # Initialize MEMTL on the first run
            memtl_model = MEMTL(mu_num, head_num)
            logging.info("Initializing and training MEMTL backbone...")
            memtl_model.set_backbone(True)
            memtl_model.set_trainable(0)
            memtl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=[custom_loss] * head_num)
            start_time = time.time()
            memtl_model.fit(X_train_all, Y_train_all[:, -mu_num:], epochs=EPOCHS, verbose=0)
            logging.info(f"MEMTL backbone training finished in {time.time() - start_time:.2f} seconds.")
            memtl_model.reset_head(0)
        else:
            logging.info(f"Adding new head to MEMTL. Total heads: {head_num}")
            memtl_model.add_head()

        # Train the newly added prediction head on its specific subset
        last_head_index = head_num - 1
        logging.info(f"Training head {last_head_index + 1} on its specific subset...")
        memtl_model.set_backbone(False)
        memtl_model.set_trainable(last_head_index)
        memtl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=[custom_loss] * head_num)
        start_time = time.time()
        memtl_model.fit(Xs[last_head_index], Ys[last_head_index][:, -mu_num:], epochs=EPOCHS, verbose=0)
        logging.info(f"Head {last_head_index + 1} training finished in {time.time() - start_time:.2f} seconds.")

        # --- Evaluation ---
        X_test, Y_test = Xs[-1], Ys[-1]

        memtl_predictions, _, _, memtl_summary, _ = _evaluate_memtl_performance(memtl_model, X_test, Y_test, mu_num)

        # Evaluate MTL
        logging.info(f">>>>>>>> Evaluating MTL on {X_test.shape[0]} test samples <<<<<<<<<<")
        
        @tf.function(jit_compile=True)
        def fast_predict_mtl(inputs):
            return mtl_model(inputs, training=False)
            
        x_tensor_mtl = tf.convert_to_tensor(X_test, dtype=tf.float32)
        _ = fast_predict_mtl(x_tensor_mtl[:1]) # Warmup
        
        start_time = time.time()
        mtl_predictions_tensor = fast_predict_mtl(x_tensor_mtl)
        mtl_predictions = mtl_predictions_tensor.numpy()
        mtl_pred_time = (time.time() - start_time) * 1000 / X_test.shape[0]
        mtl_mse_per_sample = K.sum(K.square(mtl_predictions - Y_test[:, -mu_num:])) / X_test.shape[0]
        mtl_pre_decision = np.where(mtl_predictions < DECISION_THRESHOLD, 0, 1)
        mtl_pre_class = mtl_pre_decision.dot(1 << np.arange(mtl_pre_decision.shape[-1] - 1, -1, -1))
        mtl_accuracy = accuracy_score(Y_test[:, -(mu_num + 1)], mtl_pre_class)
        logging.info(f"MTL prediction cost {mtl_pred_time:.4f}ms per sample, mse/acc = {mtl_mse_per_sample:.4f}/{mtl_accuracy:.4f}")
        mtl_summary = (f"MTL   prediction cost {mtl_pred_time:.4f}ms per sample, "
                       f"{mtl_mse_per_sample:.4f}/{mtl_accuracy:.4f}")

        # Calculate and log overall cost
        memtl_avg_cost, mtl_avg_cost, _, _ = over_all_cost(env_params, test_raw_features, memtl_predictions, mtl_predictions)
        logging.info(f"Overall cost (MEMTL/MTL): {memtl_avg_cost:.4f} / {mtl_avg_cost:.4f}")
        memtl_summary += f" overall cost {memtl_avg_cost:.4f}\n"
        mtl_summary += f" overall cost {mtl_avg_cost:.4f}\n"

        # --- Store Results and Models ---
        if head_num == head_range[1]: # On the very last run for this dataset
            output_csv_path = Path("results") / f"{dataset_path.stem}_results.csv"
            output_csv_path.parent.mkdir(exist_ok=True)
            pd.DataFrame(
                np.concatenate((test_raw_features, memtl_predictions, mtl_predictions), axis=1)
            ).to_csv(output_csv_path, index=False, header=False)
            logging.info(f"Stored final test set predictions to {output_csv_path}")

        test_results_log.append(env_summary)
        test_results_log.append(memtl_summary)
        test_results_log.append(mtl_summary)

        store_path = model_path / f"{head_num}h"
        store_model(memtl_model, str(store_path))
        logging.info(f"MEMTL model with {head_num} heads stored to {store_path}")

    return test_results_log

def MEMTL_exp():
    """
    Main function to parse config, run experiments, and save summary.
    This function handles argument parsing internally to support the original execution command.
    """
    # --- 3. Optional Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run MEMTL vs MTL comparison experiments.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        # Set a default value. This is the key change.
        default='config/MEMTL_config.yaml',
        help="Path to the YAML configuration file. Defaults to 'config/MEMTL_config.yaml'."
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    # --- 4. Robust Configuration Loading ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        return
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML config file '{config_path}': {e}")
        return

    # Safely get config values with .get() to provide helpful error messages
    file_paths = config.get('datasetPath', [])
    head_ranges = config.get('headNumRange', [])
    model_paths = config.get('modelPath', [])
    result_filename = config.get('result_file', 'final_summary.txt')
    if isinstance(result_filename, list):
        logging.warning(
            f"Configuration 'result_file' was read as a list. Using the first element: {result_filename[0]}")
        result_filename = result_filename[0]
    env_params = (config.get('F'), config.get('kappa'), config.get('P_t'),
                  config.get('P_I'), config.get('P_d'))

    if not all([file_paths, head_ranges, model_paths, all(p is not None for p in env_params)]):
        logging.error("One or more required keys are missing from the config file. "
                      "Please check datasetPath, headNumRange, modelPath, F, kappa, etc.")
        return

    all_results = []
    # --- 5. Main Experiment Loop with Progress Bar ---
    for i in tqdm(range(len(file_paths)), desc="Processing Datasets", ascii=True):
        dataset_p = Path(file_paths[i])
        model_p = Path(model_paths[i])
        model_p.mkdir(parents=True, exist_ok=True)

        logging.info(f"################### Starting Experiment on {dataset_p.name} ###################")
        results_for_dataset = train_test_store(dataset_p, head_ranges[i], env_params, model_p)
        all_results.append("".join(results_for_dataset))

    # --- 6. Write Final Summary File ---
    summary_path = Path(result_filename)
    summary_path.parent.mkdir(exist_ok=True)
    try:
        with open(summary_path, 'w') as file:
            logging.info(f"\n[Final Step] Writing summary to: {summary_path}")
            # Join all collected results with double newlines for better separation
            file.write("\n\n".join(all_results))
        logging.info("All experiments finished. Summary file written successfully.")
    except IOError as e:
        logging.error(f"Failed to write summary file to {summary_path}: {e}")

# This block is executed when you run `python -m example.MEMTL_example`
if __name__ == "__main__":
    MEMTL_exp()