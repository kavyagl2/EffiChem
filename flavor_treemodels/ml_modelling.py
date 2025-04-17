import os
import numpy as np
import optuna
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import json
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIR = "/home/raghvendra2/Molformer_Finetuning/classification_model_flavor/Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def optimize_model(trial, X_train, y_train, X_val, y_val, model_type):
    """Hyperparameter tuning using Optuna."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
    }

    if model_type == "xgboost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
    if model_type == "lightgbm":
        model = lgb.LGBMClassifier(class_weight="balanced", **params)
    elif model_type == "catboost":
        model = cb.CatBoostClassifier(auto_class_weights="Balanced", **params, verbose=0)

    model.fit(X_train, y_train)
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

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_params, embedding_name):
    """Train models with optimized hyperparameters and evaluate their performance."""
    logging.info(f"Training and evaluating models for {embedding_name} embeddings...")

    metrics_dict = {}

    # Train XGBoost
    logging.info(f"Training XGBoost for {embedding_name}...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **best_params["XGBoost"])
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

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
    lgb_model = lgb.LGBMClassifier(class_weight="balanced", **best_params["LightGBM"])  
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)

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
    cat_model = cb.CatBoostClassifier(auto_class_weights="Balanced", **best_params["CatBoost"], verbose=0)  
    cat_model.fit(X_train, y_train)
    y_pred_cat = cat_model.predict(X_test)

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

    # Save metrics
    np.save(os.path.join(OUTPUT_DIR, f"metrics_{embedding_name}.npy"), metrics_dict)
    logging.info(f"Metrics for {embedding_name} saved successfully in {OUTPUT_DIR}")

    # Save trained models
    joblib.dump(lgb_model, os.path.join(OUTPUT_DIR, f"lgb_model_{embedding_name}.pkl"))
    cat_model.save_model(os.path.join(OUTPUT_DIR, f"cat_model_{embedding_name}.cbm"))

    logging.info(f"All models for {embedding_name} saved successfully in {OUTPUT_DIR}")
