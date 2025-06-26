import pandas as pd
import logging
from embedding_processing import load_embeddings
from feature_extraction import extract_features
from ml_modelling import prepare_features, run_optimization, train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

train_path = ".../clintox_train_embed.csv"
test_path = ".../clintox_test_embed.csv"
val_path = ".../clintox_eval_embed.csv"

train_df, test_df, val_df = load_embeddings(train_path, test_path, val_path)

train_df = extract_features(train_df)
test_df = extract_features(test_df)
val_df = extract_features(val_df)

logging.info("Feature extraction completed successfully.")

embedding_types = [
            'MolFormer_Base_embeddings', 
            'MolFormer_FL_embeddings', 
            'MolFormer_WL_embeddings', 
            'ChemBERTa_77M_MTR_FL_embeddings', 
            'ChemBERTa_77M_MTR_WL_embeddings', 
            'ChemBERTa_10M_MTR_WL_embeddings', 
            'ChemBERTa_10M_MTR_FL_embeddings', 
            'ChemBERTa_77M_MLM_WL_embeddings', 
            'ChemBERTa_77M_MLM_FL_embeddings', 
            'ChemBERTa_10M_MLM_WL_embeddings', 
            'ChemBERTa_10M_MLM_FL_embeddings', 
            'ChemBERTa_5M_MTR_WL_embeddings', 
            'ChemBERTa_5M_MTR_FL_embeddings', 
            'ChemBERTa_77M_MTR_Base_embeddings', 
            'ChemBERTa_10M_MTR_Base_embeddings', 
            'ChemBERTa_10M_MLM_Base_embeddings', 
            'ChemBERTa_77M_MLM_Base_embeddings', 
            'ChemBERTa_5M_MTR_Base_embeddings'
        ]
### Loop over each classification target
classification_targets = ["CT_TOX"]

for target in classification_targets:
    for embedding in embedding_types:
        logging.info(f"Processing for {embedding} and target: {target}...")

        X_train, y_train = prepare_features(train_df, embedding, target, fit_label_encoder=True)
        X_val, y_val = prepare_features(val_df, embedding, target, fit_label_encoder=False)
        X_test, y_test = prepare_features(test_df, embedding, target, fit_label_encoder=False)

        best_params = run_optimization(X_train, y_train, X_val, y_val, embedding + "_" + target)
        train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_params, embedding + "_" + target)

        logging.info(f"Processing completed for {embedding} and target: {target}.")

logging.info("All processes completed successfully.")
