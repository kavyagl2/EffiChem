import pandas as pd
import numpy as np
import logging
from ast import literal_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_embeddings(train_path, test_path, val_path):
    """Load embeddings from CSV files and clean unnecessary columns."""
    logging.info("Loading embedding datasets...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)
    
    # Convert string embeddings back to NumPy arrays
    for df_name, df in zip(["Train", "Test", "Val"], [train_df, test_df, val_df]):
        logging.info(f"Processing {df_name} dataset...")

        embedding_columns = [
            "ChemBERTa_Base_embeddings",
            "MolFormer_Base_embeddings",
            "MolFormer_Finetuned_embeddings",
            "ChemBERTa_77M_MTR_embeddings",
            "ChemBERTa_10M_MTR_embeddings",
            "ChemBERTa_77M_MLM_embeddings",
            "ChemBERTa_10M_MLM_embeddings",
            "ChemBERTa_5M_MTR_embeddings"
        ]

        for col in embedding_columns:
            df[col] = df[col].apply(lambda x: np.array(literal_eval(x)))

    logging.info("Embedding datasets loaded successfully!")
    return train_df, test_df, val_df
