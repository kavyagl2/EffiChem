import os
import numpy as np
import optuna
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, roc_curve, auc,precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder,  label_binarize
from sklearn.utils.class_weight import compute_sample_weight
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIR = ".../Transformer_CH_Finetuned_Results"
ROC_DIR = os.path.join(OUTPUT_DIR, "ROC_Curves")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)

label_encoder = LabelEncoder()

def prepare_features(df, embedding_type, fit_label_encoder=False):
    """Prepare input features by concatenating embeddings with RDKit features (expanded columns)."""
    logging.info(f"Preparing features using {embedding_type} embeddings and RDKit features...")

    X_embed = np.stack(df[embedding_type].values)  
    rdkit_keys = ["Molecular Weight", "ALogP", "Molar Refractivity", "Heavy Atom Count", "Atom Count", 
                  "Bond Count", "Ring Count", "Aromatic Ring Count", "Saturated Ring Count", 
                  "H-Bond Acceptors", "H-Bond Donors", "Topological Polar Surface Area",
                  "Rotatable Bonds", "Formal Charge", "QED (Drug-likeness)", "Radical Electrons", "sp3 Carbon Fraction"]
    X_rdkit = np.stack(df[rdkit_keys].values)
    X_final = np.concatenate([X_embed, X_rdkit], axis=1)

    # Create column names from embeddings and RDKit features.
    embed_dim = X_embed.shape[1]
    embed_names = [f"embed_{i}" for i in range(embed_dim)]
    feature_names = embed_names + rdkit_keys
    X_final_df = pd.DataFrame(X_final, columns=feature_names)

    if fit_label_encoder:
        y = label_encoder.fit_transform(df["Canonicalized Taste"].values)
    else:
        y = label_encoder.transform(df["Canonicalized Taste"].values)
    
    logging.info(f"Feature preparation complete: {X_final_df.shape[0]} samples, {X_final_df.shape[1]} features")
    return X_final_df, y

def compute_sample_weights(y):
    return compute_sample_weight(class_weight='balanced', y=y)

def optimize_model(trial, X_train, y_train, X_val, y_val, model_type):
    """Hyperparameter tuning using Optuna."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
    }

    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            random_state=42,
            **params
        )
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(
            class_weight="balanced",
            objective="multiclass",
            metric="multi_logloss",
            random_state=42,
            **params
        )
    elif model_type == "catboost":
        model = cb.CatBoostClassifier(
            auto_class_weights="Balanced",
            loss_function="MultiClass",
            random_seed=42,
            verbose=0,
            **params
        )

    sample_weights = compute_sample_weights(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    preds = model.predict(X_val)
    
    return accuracy_score(y_val, preds)

def run_optimization(X_train, y_train, X_val, y_val, embedding_name):
    """Runs Optuna to find the best hyperparameters for each model."""
    logging.info(f"Running model optimization for {embedding_name} embeddings...")

    # Optimize XGBoost
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda trial: optimize_model(trial, X_train, y_train, X_val, y_val, "xgboost"), n_trials=10)

    # Optimize LightGBM
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lambda trial: optimize_model(trial, X_train, y_train, X_val, y_val, "lightgbm"), n_trials=10)

    # Optimize CatBoost
    study_cb = optuna.create_study(direction="maximize")
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
    """Plot and save ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    colors = {'XGBoost': 'blue', 'LightGBM': 'green', 'CatBoost': 'red'}
    
    for model_name, y_proba in y_proba_dict.items():
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
            
            # For plotting, calculate macro-average ROC curve
            fpr_dict = {}
            tpr_dict = {}
            n_classes = y_proba.shape[1]
            
            for i in range(n_classes):
                y_binary = (y_true == i).astype(int)
                fpr_dict[i], tpr_dict[i], _ = roc_curve(y_binary, y_proba[:, i])
            
            # Compute macro-average ROC curve
            all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
            
            mean_tpr /= n_classes
            
            plt.plot(all_fpr, mean_tpr, color=colors.get(model_name, 'black'), lw=2,
                    label=f'{model_name} (Macro-Avg AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {embedding_name.replace("_", " ")} - {target_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(ROC_DIR, f'roc_curve_{embedding_name}_{target_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves(y_true, y_proba_dict, embedding_name, target_name):
    """Plot and save PR curves for all models."""
    plt.figure(figsize=(10, 8))
    
    colors = {'XGBoost': 'blue', 'LightGBM': 'green', 'CatBoost': 'red'}
    n_classes = y_proba_dict['XGBoost'].shape[1]  # Assuming all models have same shape

    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    for model_name, y_proba in y_proba_dict.items():
        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            average_precision[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        
        # Compute macro-average
        all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
        mean_precision = np.zeros_like(all_recall)

        for i in range(n_classes):
            mean_precision += np.interp(all_recall, recall[i], precision[i])
        mean_precision /= n_classes

        avg_prec_score = average_precision_score(y_true_bin, y_proba, average="macro")

        plt.plot(all_recall, mean_precision, label=f'{model_name} (Avg Precision = {avg_prec_score:.3f})', 
                 lw=2, color=colors.get(model_name, 'black'))

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'PR Curves - {embedding_name.replace("_", " ")} - {target_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROC_DIR, f'pr_curve_{embedding_name}_{target_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_params, embedding_name):
    """Train models with optimized hyperparameters and evaluate their performance."""
    logging.info(f"Training and evaluating models for {embedding_name} embeddings...")

    metrics_dict = {}
    y_proba_dict = {}  # Store probabilities for ROC curves

    # Train XGBoost
    logging.info(f"Training XGBoost for {embedding_name}...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **best_params["XGBoost"], random_state=42)
    sample_weights = compute_sample_weights(y_train)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)
    y_proba_dict['XGBoost'] = y_proba_xgb

    xgb_acc = round(accuracy_score(y_test, y_pred_xgb), 4)
    xgb_f1_micro = round(f1_score(y_test, y_pred_xgb, average='micro'), 4)
    xgb_f1_macro = round(f1_score(y_test, y_pred_xgb, average='macro'), 4)
    xgb_auc = round(roc_auc_score(y_test, xgb_model.predict_proba(X_test), multi_class="ovr"), 4)
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
    lgb_model = lgb.LGBMClassifier(class_weight="balanced", random_state=42, **best_params["LightGBM"])  
    lgb_model.fit(X_train, y_train, eval_metric="multi_logloss")
    y_pred_lgb = lgb_model.predict(X_test)
    y_proba_lgb = lgb_model.predict_proba(X_test)
    y_proba_dict['LightGBM'] = y_proba_lgb

    lgb_acc = round(accuracy_score(y_test, y_pred_lgb),4)
    lgb_f1_micro = round(f1_score(y_test, y_pred_lgb, average='micro'),4)
    lgb_f1_macro = round(f1_score(y_test, y_pred_lgb, average='macro'),4)
    lgb_auc = round(roc_auc_score(y_test, lgb_model.predict_proba(X_test), multi_class="ovr"),4)
    lgb_prec, lgb_rec, _, _ = precision_recall_fscore_support(y_test, y_pred_lgb, average='macro')
    lgb_prec = round(lgb_prec, 4)
    lgb_rec = round(lgb_rec, 4)
    lgb_mcc = round(matthews_corrcoef(y_test, y_pred_lgb), 4)

    logging.info(f"LightGBM - Accuracy: {lgb_acc:.4f}, AUC: {lgb_auc:.4f}, Precision: {lgb_prec:.4f}, Recall: {lgb_rec:.4f}, F1 Macro: {lgb_f1_macro:.4f}, F1 Micro: {lgb_f1_micro:.4f}, MCC: {lgb_mcc:.4f}")
    metrics_dict["LightGBM"] = {
        "Accuracy": lgb_acc,
        "AUC": lgb_auc,
        "Precision": lgb_prec,
        "Recall": lgb_rec,
        "F1 Macro": lgb_f1_macro,
        "F1 Micro": lgb_f1_micro,
        "MCC": lgb_mcc
    }

    # Train CatBoost
    logging.info(f"Training CatBoost for {embedding_name}...")
    cat_model = cb.CatBoostClassifier(auto_class_weights="Balanced", eval_metric="MultiClass",random_seed = 42, **best_params["CatBoost"], verbose=0)  
    cat_model.fit(X_train, y_train)
    y_pred_cat = cat_model.predict(X_test)
    y_proba_cat = cat_model.predict_proba(X_test)
    y_proba_dict['CatBoost'] = y_proba_cat

    cat_acc = round(accuracy_score(y_test, y_pred_cat), 4)
    cat_f1_micro = round(f1_score(y_test, y_pred_cat, average='micro'), 4)
    cat_f1_macro = round(f1_score(y_test, y_pred_cat, average='macro'), 4)
    cat_auc = round(roc_auc_score(y_test, cat_model.predict_proba(X_test), multi_class="ovr"), 4)
    cat_prec, cat_rec, _, _ = precision_recall_fscore_support(y_test, y_pred_cat, average='macro')
    cat_prec = round(cat_prec, 4)
    cat_rec = round(cat_rec, 4)
    cat_mcc = round(matthews_corrcoef(y_test, y_pred_cat), 4)

    logging.info(f"CatBoost - Accuracy: {cat_acc:.4f}, AUC: {cat_auc:.4f}, Precision: {cat_prec:.4f}, Recall: {cat_rec:.4f}, F1 Micro: {cat_f1_micro:.4f}, F1 Macro: {cat_f1_macro:.4f}, MCC: {cat_mcc:.4f}")
    metrics_dict["CatBoost"] = {
        "Accuracy": cat_acc,
        "AUC": cat_auc,
        "Precision": cat_prec,
        "Recall": cat_rec,
        "F1 Macro": cat_f1_macro,
        "F1 Micro": cat_f1_micro,
        "MCC": cat_mcc
    }

    # Generate and save ROC curves
    target_name = "Canonicalized_Taste"  # Fixed target name for flavor classification
    embedding_clean = embedding_name  # Use embedding name as is
    
    logging.info(f"Generating ROC curves for {embedding_name}...")
    plot_roc_curves(y_test, y_proba_dict, embedding_clean, target_name)

    logging.info(f"Generating PR curves for {embedding_name}...")
    plot_pr_curves(y_test, y_proba_dict, embedding_clean, target_name)
    
    # Save prediction probabilities for later analysis
    proba_data = {
        'y_true': y_test,
        'XGBoost_proba': y_proba_xgb,
        'LightGBM_proba': y_proba_lgb,
        'CatBoost_proba': y_proba_cat
    }
    np.save(os.path.join(OUTPUT_DIR, f"probabilities_{embedding_name}.npy"), proba_data)

    # Save metrics
    np.save(os.path.join(OUTPUT_DIR, f"metrics_{embedding_name}.npy"), metrics_dict)
    logging.info(f"Metrics for {embedding_name} saved successfully in {OUTPUT_DIR}")

    # Save trained models
    joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, f"xgb_model_{embedding_name}.pkl"))
    joblib.dump(lgb_model, os.path.join(OUTPUT_DIR, f"lgb_model_{embedding_name}.pkl"))
    cat_model.save_model(os.path.join(OUTPUT_DIR, f"cat_model_{embedding_name}.cbm"))

    logging.info(f"All models and ROC curves for {embedding_name} saved successfully in {OUTPUT_DIR}")