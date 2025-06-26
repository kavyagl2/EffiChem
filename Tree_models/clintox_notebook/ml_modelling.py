import os
import numpy as np
import optuna
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIR = ".../chemberta_base_clintox_T_CH"
ROC_DIR = os.path.join(OUTPUT_DIR, "ROC_Curves")
PR_DIR = os.path.join(OUTPUT_DIR, "PR_Curves")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)
os.makedirs(PR_DIR, exist_ok=True)

label_encoder = LabelEncoder()

RANDOM_SEED = 42  # Fixing random seed for reproducibility

def prepare_features(df, embedding_type, label_column, fit_label_encoder=False):
    """Prepare input features by concatenating embeddings with RDKit features."""
    logging.info(f"Preparing features using {embedding_type} embeddings and RDKit features...")

    X_embed = np.stack(df[embedding_type].values)  
    rdkit_keys = ["Molecular Weight", "ALogP", "Molar Refractivity", "Heavy Atom Count", "Atom Count", 
                  "Bond Count", "Ring Count", "Aromatic Ring Count", "Saturated Ring Count", 
                  "H-Bond Acceptors", "H-Bond Donors", "Topological Polar Surface Area",
                  "Rotatable Bonds", "Formal Charge", "QED (Drug-likeness)", "Radical Electrons", "sp3 Carbon Fraction"]
    X_rdkit = np.stack(df[rdkit_keys].values)
    X_final = np.concatenate([X_embed, X_rdkit], axis=1)

    embed_dim = X_embed.shape[1]
    embed_names = [f"embed_{i}" for i in range(embed_dim)]
    feature_names = embed_names + rdkit_keys
    X_final_df = pd.DataFrame(X_final, columns=feature_names)

    if fit_label_encoder:
        y = label_encoder.fit_transform(df[label_column].values.astype(int))
    else:
        y = label_encoder.transform(df[label_column].values.astype(int))
    
    logging.info(f"Feature preparation complete: {X_final_df.shape[0]} samples, {X_final_df.shape[1]} features")
    return X_final_df, y


def optimize_model(trial, X_train, y_train, X_val, y_val, model_type, scale_pos_weight=None):
    """Hyperparameter tuning using Optuna with F1 score optimization for binary classification."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500)
    }

    params_XGB = {
        "max_depth": trial.suggest_int("max_depth", 2, 15),

        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),

        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),

    }

    if model_type == "xgboost":
        model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    scale_pos_weight=scale_pos_weight,
                    **params_XGB,
                    random_state=RANDOM_SEED
                )
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(
                    objective='binary',
                    class_weight="balanced",
                    metric='binary_logloss',
                    random_state=RANDOM_SEED,
                    **params
                )
    elif model_type == "catboost":
        # Set eval metric and random seed explicitly
        model = cb.CatBoostClassifier(
                    auto_class_weights="Balanced",
                    eval_metric='Logloss',
                    random_seed=RANDOM_SEED,
                    verbose=0,
                    **params
                )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    # Optimize based on F1-score (better for imbalanced data)
    f1 = f1_score(y_val, preds)
    return f1


def run_optimization(X_train, y_train, X_val, y_val, embedding_name):
    """Runs Optuna to find the best hyperparameters for each model, optimizing F1-score."""
    logging.info(f"Running model optimization for {embedding_name} embeddings...")

    # Calculate scale_pos_weight for XGBoost if binary classification (imbalance handling)
    num_class_0 = np.sum(y_train == 0)
    num_class_1 = np.sum(y_train == 1) 

    """Here we have to remember one thing that is, in clintox dataset we have :
    1. CT_TOX: here class 0 is in majority and class 1 is in minority
    2. FDA_APPROVED: here class 1 is in majority and class 0 is in minority
    and, 
    scale_pos_weight = count of majority class / count of minority class
    so we are adding the condition to check which class is majority and accordingly calculating scale_pos_weight"""

    if num_class_0 > num_class_1:
        scale_pos_weight = num_class_0 / num_class_1
        logging.info(f"Class 0 is majority: scale_pos_weight = {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = num_class_1 / num_class_0
        logging.info(f"Class 1 is majority: scale_pos_weight = {scale_pos_weight:.2f}")

    # Optimize XGBoost
    study_xgb = optuna.create_study(direction="maximize")  # maximize F1-score
    study_xgb.optimize(lambda trial: optimize_model(trial, X_train, y_train, X_val, y_val, "xgboost", scale_pos_weight), n_trials=10)

    # Optimize LightGBM
    study_lgb = optuna.create_study(direction="maximize")  # maximize F1-score
    study_lgb.optimize(lambda trial: optimize_model(trial, X_train, y_train, X_val, y_val, "lightgbm"), n_trials=10)

    # Optimize CatBoost
    study_cb = optuna.create_study(direction="maximize")  # maximize F1-score
    study_cb.optimize(lambda trial: optimize_model(trial, X_train, y_train, X_val, y_val, "catboost"), n_trials=10)

    # Save best hyperparameters
    best_params = {
        "XGBoost": study_xgb.best_params,
        "LightGBM": study_lgb.best_params,
        "CatBoost": study_cb.best_params
    }     
    with open(os.path.join(OUTPUT_DIR, f"best_params_{embedding_name}.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    logging.info(f"Best parameters for {embedding_name} saved to {OUTPUT_DIR}")
    return best_params


def plot_roc_curves(y_true, y_proba_dict, embedding_name, target_name):
    """Plot and save ROC curves for binary classification only."""
    plt.figure(figsize=(10, 8))
    
    colors = {'XGBoost': 'blue', 'LightGBM': 'green', 'CatBoost': 'red'}
    
    for model_name, y_proba in y_proba_dict.items():
        # Binary classification: use probability of positive class
        y_scores = y_proba[:, 1]  # Probability of positive class
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors.get(model_name, 'black'), lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {embedding_name.replace("_", " ")} - {target_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ROC_DIR, f'roc_curve_{embedding_name}_{target_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves(y_true, y_proba_dict, embedding_name, target_name):
    """Plot and save Precision-Recall curves for binary classification only."""
    plt.figure(figsize=(10, 8))
    
    colors = {'XGBoost': 'blue', 'LightGBM': 'green', 'CatBoost': 'red'}
    
    for model_name, y_proba in y_proba_dict.items():
        y_scores = y_proba[:, 1]  # Probability of positive class

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        plt.plot(recall, precision, color=colors.get(model_name, 'black'), lw=2,
                 label=f'{model_name} (AP = {ap:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - {embedding_name.replace("_", " ")} - {target_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    pr_curve_path = os.path.join(PR_DIR, f'pr_curve_{embedding_name}_{target_name}.png')
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_params, embedding_name):
    """Train models with optimized hyperparameters and evaluate their performance for binary classification."""
    logging.info(f"Training and evaluating models for {embedding_name} embeddings...")

    metrics_dict = {}
    y_proba_dict = {}  # Store probabilities for ROC curves

    # Train XGBoost
    logging.info(f"Training XGBoost for {embedding_name}...")
    xgb_model = xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=RANDOM_SEED,
                    **best_params["XGBoost"]
                )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)
    y_proba_dict['XGBoost'] = y_proba_xgb

    y_proba_xgb_pos = y_proba_xgb[:, 1]  # Probability of positive class

    xgb_acc = round(accuracy_score(y_test, y_pred_xgb), 4)
    xgb_f1_micro = round(f1_score(y_test, y_pred_xgb, average='micro'), 4)
    xgb_f1_macro = round(f1_score(y_test, y_pred_xgb, average='macro'), 4)
    xgb_auc = round(roc_auc_score(y_test, y_proba_xgb_pos), 4)
    xgb_prec, xgb_rec, _, _ = precision_recall_fscore_support(y_test, y_pred_xgb, average='macro')
    xgb_prec = round(xgb_prec, 4)
    xgb_rec = round(xgb_rec, 4)
    xgb_mcc = round(matthews_corrcoef(y_test, y_pred_xgb), 4)

    logging.info(f"XGBoost - Accuracy: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}, Precision: {xgb_prec:.4f}, Recall: {xgb_rec:.4f}, F1 Micro: {xgb_f1_micro:.4f}, F1 Macro: {xgb_f1_macro:.4f}, MCC: {xgb_mcc:.4f}")
    metrics_dict["XGBoost"] = {
        "Accuracy": xgb_acc,
        "AUC": xgb_auc,
        "Precision": xgb_prec,
        "Recall": xgb_rec,
        "F1 Micro": xgb_f1_micro,
        "F1 Macro": xgb_f1_macro,
        "MCC": xgb_mcc
    }

    # Train LightGBM
    logging.info(f"Training LightGBM for {embedding_name}...")
    lgb_model = lgb.LGBMClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        **best_params["LightGBM"]
                    )
    lgb_model.fit(X_train, y_train, eval_metric='binary_logloss')
    y_pred_lgb = lgb_model.predict(X_test)
    y_proba_lgb = lgb_model.predict_proba(X_test)
    y_proba_dict['LightGBM'] = y_proba_lgb

    y_proba_lgb_pos = y_proba_lgb[:, 1]

    lgb_acc = round(accuracy_score(y_test, y_pred_lgb), 4)
    lgb_f1_micro = round(f1_score(y_test, y_pred_lgb, average='micro'), 4)
    lgb_f1_macro = round(f1_score(y_test, y_pred_lgb, average='macro'), 4)
    lgb_auc = round(roc_auc_score(y_test, y_proba_lgb_pos), 4)
    lgb_prec, lgb_rec, _, _ = precision_recall_fscore_support(y_test, y_pred_lgb, average='macro')
    lgb_prec = round(lgb_prec, 4)
    lgb_rec = round(lgb_rec, 4)
    lgb_mcc = round(matthews_corrcoef(y_test, y_pred_lgb), 4)

    logging.info(f"LightGBM - Accuracy: {lgb_acc:.4f}, AUC: {lgb_auc:.4f}, Precision: {lgb_prec:.4f}, Recall: {lgb_rec:.4f}, F1 Micro: {lgb_f1_micro:.4f}, F1 Macro: {lgb_f1_macro:.4f}, MCC: {lgb_mcc:.4f}")
    metrics_dict["LightGBM"] = {
        "Accuracy": lgb_acc,
        "AUC": lgb_auc,
        "Precision": lgb_prec,
        "Recall": lgb_rec,
        "F1 Micro": lgb_f1_micro,
        "F1 Macro": lgb_f1_macro,
        "MCC": lgb_mcc
    }

    # Train CatBoost
    logging.info(f"Training CatBoost for {embedding_name}...")
    cb_model = cb.CatBoostClassifier(
                        auto_class_weights="Balanced",
                        eval_metric='Logloss',
                        random_seed=RANDOM_SEED,
                        verbose=0,
                        **best_params["CatBoost"]
                    )
    cb_model.fit(X_train, y_train)
    y_pred_cb = cb_model.predict(X_test)
    y_proba_cb = cb_model.predict_proba(X_test)
    y_proba_dict['CatBoost'] = y_proba_cb

    y_proba_cb_pos = y_proba_cb[:, 1]

    cb_acc = round(accuracy_score(y_test, y_pred_cb), 4)
    cb_f1_micro = round(f1_score(y_test, y_pred_cb, average='micro'), 4)
    cb_f1_macro = round(f1_score(y_test, y_pred_cb, average='macro'), 4)
    cb_auc = round(roc_auc_score(y_test, y_proba_cb_pos), 4)
    cb_prec, cb_rec, _, _ = precision_recall_fscore_support(y_test, y_pred_cb, average='macro')
    cb_prec = round(cb_prec, 4)
    cb_rec = round(cb_rec, 4)
    cb_mcc = round(matthews_corrcoef(y_test, y_pred_cb), 4)

    logging.info(f"CatBoost - Accuracy: {cb_acc:.4f}, AUC: {cb_auc:.4f}, Precision: {cb_prec:.4f}, Recall: {cb_rec:.4f}, F1 Micro: {cb_f1_micro:.4f}, F1 Macro: {cb_f1_macro:.4f}, MCC: {cb_mcc:.4f}")
    metrics_dict["CatBoost"] = {
        "Accuracy": cb_acc,
        "AUC": cb_auc,
        "Precision": cb_prec,
        "Recall": cb_rec,
        "F1 Micro": cb_f1_micro,
        "F1 Macro": cb_f1_macro,
        "MCC": cb_mcc
    }

    # Save ROC and PR curves

    target_name = embedding_name.split('_')[-1]  # Extract target name from embedding_name
    embedding_clean = '_'.join(embedding_name.split('_')[:-1])  # Extract embedding name without target

    plot_roc_curves(y_test, y_proba_dict, embedding_clean, target_name)
    plot_pr_curves(y_test, y_proba_dict, embedding_clean, target_name)

    proba_data = {
    'y_true': y_test,
    'XGBoost_proba': y_proba_xgb,
    'LightGBM_proba': y_proba_lgb,
    'CatBoost_proba': y_proba_cb
    }

    np.save(os.path.join(OUTPUT_DIR, f"probabilities_{embedding_name}.npy"), proba_data) # Save probabilities for ROC/PR curves

    np.save(os.path.join(OUTPUT_DIR, f"metrics_{embedding_name}.npy"), metrics_dict) #Metrics dictionary
 
    # Save models
    joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, f"xgb_model_{embedding_name}.pkl"))
    joblib.dump(lgb_model, os.path.join(OUTPUT_DIR, f"lgb_model_{embedding_name}.pkl"))
    cb_model.save_model(os.path.join(OUTPUT_DIR, f"cat_model_{embedding_name}.cbm"))
    
    logging.info(f"All models and ROC, PR curves for {embedding_name} saved successfully in {OUTPUT_DIR}")
