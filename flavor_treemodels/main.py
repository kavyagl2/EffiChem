import pandas as pd
import logging
from embedding_processing import load_embeddings
from feature_extraction import extract_features
from ml_modelling import prepare_features, run_optimization, train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

train_path = "/home/raghvendra2/Molformer_Finetuning/flavor_output_embeddings/all_models/fart_train_embed.csv"
test_path = "/home/raghvendra2/Molformer_Finetuning/flavor_output_embeddings/all_models/fart_test_embed.csv"
val_path = "/home/raghvendra2/Molformer_Finetuning/flavor_output_embeddings/all_models/fart_eval_embed.csv"

train_df, test_df, val_df = load_embeddings(train_path, test_path, val_path)

train_df = extract_features(train_df)
test_df = extract_features(test_df)
val_df = extract_features(val_df)

logging.info("Feature extraction completed successfully.")

# Process both embeddings
embedding_types = [
    "ChemBERTa_Base_embeddings",
    "MolFormer_Base_embeddings",
    "ChemBERTa_Finetuned_embeddings",
    "MolFormer_Finetuned_embeddings",
    "ChemBERTa_77M_MTR_embeddings",
    "ChemBERTa_10M_MTR_embeddings",
    "ChemBERTa_77M_MLM_embeddings",
    "ChemBERTa_10M_MLM_embeddings",
    "ChemBERTa_5M_MTR_embeddings"
]

for embedding in embedding_types:
    logging.info(f"Processing for {embedding}...")

    X_train, y_train = prepare_features(train_df, embedding, fit_label_encoder=True)
    X_val, y_val = prepare_features(val_df, embedding, fit_label_encoder=False)
    X_test, y_test = prepare_features(test_df, embedding, fit_label_encoder=False)

    best_params = run_optimization(X_train, y_train, X_val, y_val, embedding)
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_params, embedding)

    logging.info(f"Processing completed for {embedding}.")
    
logging.info("All processes completed successfully.")
