###############################################################################
#                      MLjar - Supevised Training Script                      #
###############################################################################
import pandas as pd
import numpy as np
import os
import time
import logging
import argparse
import sys
from pathlib import Path
import shutil

from supervised.automl import AutoML
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score,
                             accuracy_score,
                             average_precision_score,
                             balanced_accuracy_score,
                             precision_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             f1_score,
                             roc_curve)

logger = logging.getLogger(__file__)

_DEFAULT_LOG_FILENAME = '_output_MLjar.log'
_LOGGER_LEVEL = 'INFO'

# FUNCTIONS
# =============================================================================
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_seconds(total_seconds):
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    
    return int(hours), int(minutes), int(seconds)

def save_figures(df_orig: pd.DataFrame, y: pd.Series,
                 base_path, preset: str) -> pd.DataFrame:

    logger.info('Creating Figures')
    path_models = os.path.join(base_path, preset)
    path_results = os.path.join(path_models, 'Results')

    df = df_orig.copy()

    cm_00, cm_01 = [], []
    cm_10, cm_11= [], []
    auc_list = []

    for (_, ii_test), n_fold in zip(cv_outer.split(df, y), range(cv_outer.get_n_splits(df, y))):
        X_test = df.iloc[ii_test, :]
        y_test = y.iloc[ii_test]

        # Load AutoGluon Models
        path_model = os.path.join(path_models, 'Model_' + str(n_fold))
        predictor = AutoML(results_path=path_model)

        proba = predictor.predict_proba(X_test)[:, 1]
        preds = predictor.predict(X_test)

        auc_list.append(roc_auc_score(y_test, proba))

        # Compute ROC AUC        
        fpr, tpr, _ = roc_curve(y_test, proba)

        # Compute Confusion Matrix                    
        cm = confusion_matrix(y_test, preds)
        cm_00.append(cm[0,0]); cm_01.append(cm[0,1])
        cm_10.append(cm[1,0]); cm_11.append(cm[1,1])

        plt.step(fpr, tpr, lw=1.2,
                label='ROC %d (AUC = %0.3f)' % (n_fold, auc_list[n_fold]), alpha=0.8) 

    
    # ================================================================================================    
    # ROC CURVES
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.2, color='r',
        label='No Skill', alpha=.7)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Ensemble ROC Curves', fontsize=12)
    plt.legend(loc="lower right", fontsize=8)

    plt.savefig(os.path.join(path_results, 'RocCurve_Test.png'), dpi=400)
    # plt.show()
    plt.close()

    # ================================================================================================

    # CONFUSION MATRIX
    cm_mean = np.array([np.sum(cm_00), np.sum(cm_01),
                        np.sum(cm_10), np.sum(cm_11)]).reshape(2,2)
    row_sums = cm_mean.sum(axis=1)
    cm_mean_norm = cm_mean / row_sums[:, np.newaxis]

    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].set_title('Average', fontsize=12) # fontweight='bold'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_mean, display_labels=['No', 'Yes'])
    disp.plot(ax=ax[0], cmap='Blues')
    disp.im_.set_clim(vmin=0, vmax=cm_mean.sum())

    ax[1].set_title('Average Normalized', fontsize=12) # fontweight='bold',
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_mean_norm, display_labels=['No', 'Yes'])
    disp.plot(ax=ax[1], cmap='Blues')
    disp.im_.set_clim(vmin=0, vmax=1)
    
    fig.suptitle(f'Confusion Matrix', fontsize=16)
    fig.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(path_results, 'CMatrix_Test.png'), dpi=400) 
    plt.close()
    
def train_autoMLjar_presets(X: pd.DataFrame, y: pd.Series, cv_outer, base_path,
                            presets: list[str] = ['Explain', 'Perform', 'Compete', 'Optuna']):

    def calculate_metrics(model, X, y) -> dict:
        # Predictions
        y_hat_proba = model.predict_proba(X)[:, 1]
        y_hat = model.predict(X)

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()

        scores = {
            'AUC': roc_auc_score(y, y_hat_proba),
            'PR_AUC': average_precision_score(y, y_hat_proba),
            'F1': f1_score(y, y_hat),
            'Accuracy': accuracy_score(y, y_hat),
            'BalAcc': balanced_accuracy_score(y, y_hat),
            'Sensitivity_Recall': tp / (tp + fn),
            'Specificity': tn / (tn + fp),
            'Precision': precision_score(y, y_hat),
            }

        return scores
    
    preset_times = []
    for preset in presets:
        logger.info(f'Start Training Preset: {preset}')
        logger.info('##########' * 8)
        start_time = time.time()

        # Create Directories
        path_preset = os.path.join(base_path, preset)
        path_results = os.path.join(path_preset, 'Results')
        ensure_directory_exists(path_preset)
        ensure_directory_exists(path_results)

        metrics_train, metrics_test = [], []
        fold_times = []

        for (ii_train, ii_test), n_fold in zip(cv_outer.split(X, y), range(cv_outer.get_n_splits(X, y))):
            # logger.info(f'Star Fold = {n_fold}')
            X_train, X_test = X.iloc[ii_train, :], X.iloc[ii_test, :]
            y_train, y_test = y.iloc[ii_train], y.iloc[ii_test]
            _, _ = X_train.pop('PatientID'), X_test.pop('PatientID')
            
            # Measure Training Time by Fold
            start_time_fold = time.time()
            path_model = os.path.join(path_preset, 'Model_' + str(n_fold))

            predictor = AutoML(eval_metric='auc', mode=preset, results_path=path_model, explain_level=1)
            predictor.fit(X_train, y_train)

            metrics_train.append(calculate_metrics(predictor, X_train, y_train))
            metrics_test.append(calculate_metrics(predictor, X_test, y_test))

            end_time_fold = time.time()
            total_time_fold = convert_seconds(end_time_fold - start_time_fold)
            logger.info(f"PRESET = {preset} | Fold {n_fold} execution time: {total_time_fold[0]} h. {total_time_fold[1]} m. {total_time_fold[2]} s.")
            fold_times.append({'Fold': n_fold, 'Time (s)': (end_time_fold - start_time_fold), 'Time': total_time_fold})
            
        # Save Data
        logger.info('Saving Results')
        metrics_train, metrics_test = pd.DataFrame(metrics_train), pd.DataFrame(metrics_test)
        metrics_train.to_csv(os.path.join(path_preset, 'Results', 'Metrics_Train.csv'), index=True)
        metrics_test.to_csv(os.path.join(path_preset, 'Results', 'Metrics_Test.csv'), index=True)
        pd.DataFrame(fold_times).to_csv(os.path.join(path_preset, 'Results', 'Fold_Times.csv'), index=True)

        # Measure Total Time
        end_time = time.time()
        total_time = convert_seconds(end_time - start_time)
        logger.info(f"PRESET = {preset} | Total execution time: {total_time[0]} h. {total_time[1]} m. {total_time[2]} s.")
        preset_times.append({'Preset': preset, 'Time (s)': (end_time - start_time), 'Time': total_time})
    
    logger.info('Saving Preset Times')
    pd.DataFrame(preset_times).to_csv(os.path.join(base_path, 'Preset_Times.csv'), index=True)
    logger.info('END TRAINING')

def compute_average_metrics(base_path):

    def process_df(path: str):
        metrics = pd.read_csv(path, index_col=0)
        metrics = pd.concat([metrics.mean(), metrics.std()], axis=1).round(3).\
            rename({0: 'Mean', 1: 'Std'}, axis=1)
        return metrics
    
    logger.info(f'Calculate Average Metrics')
    # Obtain presets
    presets = os.listdir(base_path)
    presets = [preset for preset in presets if preset in ['Explain', 'Perform', 'Compete', 'Optuna']]
    
    for preset in presets:
        logger.info(f'Calculating Average Metrics | PRESET: {preset}')
        path_preset = os.path.join(base_path, preset)

        # Compute Mean and Std
        metrics_train = process_df(os.path.join(path_preset, 'Results', 'Metrics_Train.csv'))
        metrics_test = process_df(os.path.join(path_preset, 'Results', 'Metrics_Test.csv'))
        
        # Save
        metrics_train.to_csv(os.path.join(path_preset, 'Results', 'Avg_Metrics_Train.csv'))
        metrics_test.to_csv(os.path.join(path_preset, 'Results', 'Avg_Metrics_Test.csv'))
    logger.info(f'END AVERAGE METRICS')


# logger
# =============================================================================
parser = argparse.ArgumentParser(description='Classification models.')
parser.add_argument("-i", "--input", help="path to patients folder.")
args = parser.parse_args()
path_input = args.input

# Configure logger
message_format = ('%(asctime)s | %(module)-16s | %(levelname)-8s | '
                '%(message)s')
date_format = '%Y/%m/%d %H:%M:%S'
formatter = logging.Formatter(message_format, datefmt=date_format)

log_level = os.environ.get('LOGLEVEL', _LOGGER_LEVEL).upper()

logging.basicConfig(level=log_level, format=message_format,
                    datefmt=date_format, stream=sys.stdout)

root_logger = logging.getLogger()
file_handler = logging.FileHandler(Path(path_input) / _DEFAULT_LOG_FILENAME, mode='w')
file_handler.setFormatter(formatter)
file_handler.setLevel(log_level)
root_logger.addHandler(file_handler)

logger.setLevel(logging.INFO)


base_path = os.path.join(path_input, 'MLjar_Models')

if os.path.exists(base_path):
    shutil.rmtree(base_path)

ensure_directory_exists(base_path)


# MAIN
# =============================================================================
SEED = 7
outer_splits = 5
cv_outer = StratifiedKFold(n_splits=outer_splits, random_state=SEED, shuffle=True)

# Load Data
X = pd.read_csv(os.path.join(path_input, 'data.csv'))

y = X.iloc[:, -1]
X = X.iloc[:, :-1]

try:
    train_autoMLjar_presets(X, y, cv_outer, base_path)
    compute_average_metrics(base_path)

    for preset in ['Explain', 'Perform', 'Compete', 'Optuna']:
        try:
            save_figures(X, y, base_path, preset)
        except Exception as e:
            logger.error(f'Figures not saved for {preset}')
            logger.error(e)

    logger.info('FINAL END')
except Exception as e:
    logger.error(e)
