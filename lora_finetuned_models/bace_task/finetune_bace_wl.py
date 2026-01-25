# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: effichem
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Hyperparameter tuning using WandB

# %%
#Importing Libraries
import wandb
import evaluate
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import load
from datasets import Dataset
import numpy as np
import pandas as pd
import os
import gc
from scipy.special import softmax
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,matthews_corrcoef

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dotenv import load_dotenv
load_dotenv()
wandb_api_key=os.getenv("WANDB_API_KEY")

#Set up WANDB Key
if wandb_api_key is not None:
    print(f"API Key loaded successfully! Key snippet: {wandb_api_key[:5]}...")
else:
    print("API Key not found. Please check your .env file and environment setup.")

wandb.login(key=wandb_api_key)


# %%
# Load train and valid datasets
def data_load():
    train_bace=pd.read_csv('bace_datasets/train_clean.csv')
    val_bace=pd.read_csv('bace_datasets/valid_clean.csv')

    return train_bace, val_bace


# %%
#Process the data into tensor format
def data_prep(data_process,tokenizer):

    smiles_list = data_process['Standardized SMILES'].tolist()
    tokenized=tokenizer(smiles_list)


    dataset = Dataset.from_dict(tokenized)
    labels = data_process['Class'].tolist()

    dataset = dataset.add_column("labels", labels)

    return dataset

# %%
from peft import LoraConfig, get_peft_model, PeftModel

def lora_config(r,lora_alpha,dropout):

    lora_config = LoraConfig(
        task_type="SEQ_CLS",  # Sequence classification task
        r=r,
        lora_alpha=lora_alpha,
        target_modules='all-linear',
        lora_dropout=dropout
    )

    return lora_config


# %% [markdown]
# ### Weighted Loss Function

# %%
#Calculate class weights
def class_weights_calculation(train_dataset):

        # Calculate class weights based on the distribution of labels
        class_weights = [1 - (train_dataset['labels'].count(0) / len(train_dataset['labels'])),
                        1 - (train_dataset['labels'].count(1) / len(train_dataset['labels']))]
        return torch.from_numpy(np.array(class_weights)).float()

#Create custom weighted loss trainer
class WeightedLossTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Extract labels
        labels = inputs.get("labels")

        class_weights= class_weights_calculation(self.train_dataset)
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_func = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

        # compute loss
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# %% [markdown]
# ### Focal Loss Function

# %%
#focal loss computation
import torch.nn.functional as F
import torch

def focal_loss(inputs, targets, alpha=1, gamma=2):
    log_prob = F.log_softmax(inputs, dim=-1)
    prob = torch.exp(log_prob)  # Convert log probabilities back to normal probabilities

    targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
    pt = torch.sum(prob * targets_one_hot, dim=-1)  # Get probability of the true class

    focal_loss = -alpha * (1 - pt) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)

    return focal_loss.mean()

#Create custom focal loss trainer
class FocalLossTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


# %%
#Evaluation for validation and test sets
from evaluate import load
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,matthews_corrcoef

accuracy_metric = load("accuracy")
mcc_metric= load("matthews_correlation")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = softmax(logits, axis=1)[:, 1]  # Get probabilities for class 1
    predictions = np.argmax(logits, axis=1)  # Choose the most likely class


    mcc = matthews_corrcoef(labels, predictions)

    return {
        "eval_mcc_metric": mcc,
        "Accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "AUC-ROC": roc_auc_score(labels, probabilities),  # AUC-ROC requires probabilities
        "Precision": precision_score(labels, predictions,average="macro",zero_division=0),
        "Recall": recall_score(labels, predictions,average="macro",zero_division=0),
        "F1-score": f1_score(labels, predictions,average="macro"),
        "F1-micro": f1_score(labels, predictions,average="micro")
    }

# %%
import re

#Perform a sweep using WANDb
sweep_config = {
"name": "BACE_Hyperparameter_Tuning",
"method": "bayes",
"metric": {
    "goal": "maximize",
    "name": "eval/mcc_metric"},
"parameters": {"lr": {
        "distribution": "uniform",
        "min": 1e-5,
        "max": 2e-3},
    "r": {"values": [4,8,16,32,64]},
    "lora_alpha": {"values": [8,16,32,64,128]},
    "dropout": {"values": [0.0,0.1,0.2] },

    "optimizer": {"value": ["adamw"]}}
}

model_list= ["DeepChem/ChemBERTa-77M-MLM",
             "DeepChem/ChemBERTa-77M-MTR",
             "ibm/MoLFormer-XL-both-10pct"]

best_configs={}

def safe_model_name(name1):
        return re.sub(r"[^a-zA-Z0-9]", "__", name1)

for model_name in model_list:

    sweep_id = wandb.sweep(sweep_config, project="huggingface")

    model_id_clean = safe_model_name(model_name)
    print(f"Running sweep for model: {model_id_clean}")

    def run_training():
        print(f"Running training for model: {model_id_clean}")
        # Initialize W&B with sweep
        run = wandb.init(project="BACE_Hyperparameter_Tuning")
        config = run.config

        print(f"Model ID cleaned: {model_id_clean}")
        run_id = wandb.run.id

        # Define unique output folders
        save_dir = f"./WL_Loss/{model_id_clean}/{run_id}"
        logging_dir = f"./WL_Loss/{model_id_clean}/{run_id}"
        os.makedirs(save_dir, exist_ok=True)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            trust_remote_code=True
        )

        # Load and preprocess data
        train_data, val_data = data_load()
        training_data = data_prep(train_data, tokenizer)
        validation_data = data_prep(val_data, tokenizer)

        # Apply LoRA
        peft_config = lora_config(config.r, config.lora_alpha, config.dropout)
        lora_model = get_peft_model(model, peft_config)
        lora_model.print_trainable_parameters()

        # Define training args
        training_args = TrainingArguments(
            output_dir=save_dir,
            eval_strategy="epoch",
            learning_rate=config.lr,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            num_train_epochs=20,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir=logging_dir,
            logging_strategy="steps",
            logging_steps=500,
            report_to="wandb",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_mcc_metric"
        )

        accuracy_metric = load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probabilities = softmax(logits, axis=1)[:, 1]
            predictions = np.argmax(logits, axis=1)
            mcc = matthews_corrcoef(labels, predictions)

            return {
                "eval_mcc_metric": mcc,
                "Accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
                "AUC-ROC": roc_auc_score(labels, probabilities),
                "Precision": precision_score(labels, predictions, average="macro", zero_division=0),
                "Recall": recall_score(labels, predictions, average="macro", zero_division=0),
                "F1-score": f1_score(labels, predictions, average="macro"),
                "F1-micro": f1_score(labels, predictions, average="micro")
            }

        # Train with weigted loss trainer
        trainer = WeightedLossTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=training_data,
            eval_dataset=validation_data,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        '''
        # train with focal loss trainer
        trainer = FocalLossTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=training_data,
            eval_dataset=validation_data,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )'''

        trainer.train()
        trainer.save_model(save_dir)
        print(f"Model saved to {save_dir}")
        print(f"Training completed for model: {model_name}")
        del trainer
        torch.cuda.empty_cache()

        wandb.finish()

    #Run a wandb agent
    wandb.agent(sweep_id, function=run_training, count=20)

    #Perform a sweep
    api = wandb.Api()
    sweep = api.sweep(f"huggingface/{sweep_id}")

    #Find the run with the best mcc
    runs_with_mcc = [run for run in sweep.runs if 'eval/mcc_metric' in run.summary_metrics]

    def get_best_epoch_mcc(run):
        history = run.scan_history(keys=["eval/mcc_metric"])
        val = max((row["eval/mcc_metric"] for row in history if row.get("eval/mcc_metric") is not None),default=float("-inf"))
        print(f"Best eval/mcc_metric for {run.id} was: {val}")
        return val

    if runs_with_mcc:
        # Sort by mcc in descending order (maximize)
        #best_run = sorted(runs_with_mcc, key=lambda run: -run.summary_metrics['eval/mcc_metric'])[0]
        best_run = max(runs_with_mcc, key=get_best_epoch_mcc)

    else:
        raise ValueError("No runs found with 'eval/mcc_metric' metric.")

    best_hyperparameters = best_run.config
    print(f"Best hyperparameters: {best_hyperparameters}")
    print("Completed sweep for model: ",model_id_clean)

    best_configs[model_id_clean] = best_run

    del api, sweep, sweep_id, runs_with_mcc, model_id_clean
    gc.collect()

print(best_configs)

# %%

# Map your folder names to the base HuggingFace model names
MODEL_NAME_MAP = {
    "DeepChem__ChemBERTa__77M__MLM": "DeepChem/ChemBERTa-77M-MLM",
    "DeepChem__ChemBERTa__77M__MTR": "DeepChem/ChemBERTa-77M-MTR",
    "ibm__MoLFormer__XL__both__10pct": "ibm/MoLFormer-XL-both-10pct",
}

#Load the test data
test_data=pd.read_csv('bace_datasets/test_clean.csv')

#Path to model root
models_root_dir = "./WL_Loss/" #add model saved path
output_dir = "./test_results_bace/"

#Essential args for evaluation
eval_args = TrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=64,
    report_to="none",
    disable_tqdm=True,
)

all_test_results = []
for key in MODEL_NAME_MAP.keys():

    model_folder = models_root_dir+key
    print("Model folder: ",model_folder)

    best_run = best_configs[key]
    checkpoint_path = model_folder+"/"+best_run.id
    print("Best PEFT folder: ",checkpoint_path)

    hf_model_name = MODEL_NAME_MAP[key]
    print(f"Using base model: {hf_model_name}")

    # Load tokenizer and base model for the model type
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        num_labels=2,
        problem_type="single_label_classification",
        trust_remote_code=True
    )

    # Create the SMILES dataset in ingestible format
    smiles_test = test_data['Standardized SMILES'].tolist()
    test_tokenized = tokenizer(smiles_test)
    test_dataset = Dataset.from_dict(test_tokenized)
    test_labels = test_data['Class'].tolist()
    test_dataset = test_dataset.add_column("labels", test_labels)

    # Load the adapter checkpoint
    adapter_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    adapter_model.eval()

    # Evaluation
    trainer = Trainer(
        model=adapter_model,
        args=eval_args,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    print(f"\n Evaluating {checkpoint_path}")

    test_results = trainer.evaluate()
    test_results["best_checkpoint"]=checkpoint_path
    all_test_results.append(test_results)

    # Save Base Model with LoRA weights
    final_adapter_model = adapter_model.merge_and_unload()
    save_path = "./weighted_loss_BACE/"+key+"_LoRA_Finetuned/"
    final_adapter_model.save_pretrained(save_path)

    command1 = "rm -rf "+model_folder
    os.system(command1)

all_test_results_df = pd.DataFrame(all_test_results)
all_test_results_df.columns = ["MCC","LOSS","Time","ACC","AUROC","PREC","REC","F1","F1-MICRO","RUNTIME","SAMPLE_PER_SECOND","STEPS_PER_SECOND","BEST_MODEL"]
print(all_test_results_df)

all_test_results_df.to_csv(output_dir+"/weighted_loss_test_results.csv",header="infer",index=False)

command2="rm -rf ./raghvendra5688 ./wandb"
os.system(command2)



# %%
