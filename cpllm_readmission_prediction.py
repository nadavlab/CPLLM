import pickle
from random import randint

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve
from transformers import TrainingArguments, AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding, Trainer
import numpy as np
from datasets import Dataset

EPOCHS = 4
MAX_LENGTH = 2048
OUTPUT_DIR = "/tmp/pycharm_project_410/"
NUM_LABELS = 2
MODEL_ID = "meta-llama/Llama-2-13b-hf"

# DATA
# Each pickle file contains a list of examples. Each example is a dictionary in the following format:
# {"visit_id": "324304",
#  "patient_id": "34533",
#  "conditions": [["Pleurisy; pneumothorax; pulmonary collapse",
#                  "Shock"]],
#  "procedures": [["first procedure",
#                  "second procedure"]],
#  "drugs": [["NOREPINEPHRINE 8 MG in 250mL NS",
#             "LEVOFLOXACIN 750 mg in D5W 150mL",
#             "FAMOTIDINE 20 MG/2 ML SDV INJ"]],
#  "label": 1}

train_pickle_file_path = '/sise/nadav-group/nadavrap-group/ofir/pyhealth_data/readmission/eicu_crd_readmission_prediction_with_descriptions_train.pickle'
validation_pickle_file_path = '/sise/nadav-group/nadavrap-group/ofir/pyhealth_data/readmission/eicu_crd_readmission_prediction_with_descriptions_validation.pickle'
test_pickle_file_path = '/sise/nadav-group/nadavrap-group/ofir/pyhealth_data/readmission/eicu_crd_readmission_prediction_with_descriptions_test.pickle'


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID,
                                                           config=AutoConfig.from_pretrained(MODEL_ID,
                                                                                             trust_remote_code=True,
                                                                                             num_labels=NUM_LABELS),
                                                           trust_remote_code=True,
                                                           load_in_8bit=True).bfloat16()
print(model)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)  # TODO: RETURN IT BACK FOR Q.

print_trainable_parameters(model)
model = get_peft_model(model, config)
print_trainable_parameters(model)

model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))


def compute_metrics(p):
    # Apply softmax to model predictions and save them in p.predictions
    p.predictions = torch.from_numpy(p.predictions)
    p.predictions = torch.softmax(p.predictions, dim=1)

    preds = np.argmax(p.predictions.cpu().numpy(), axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1-score for both classes
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    precision1, recall1, thresholds = precision_recall_curve(labels, p.predictions[:, 1], pos_label=1)
    auc_precision_recall = auc(recall1, precision1)
    positive_confidences = p.predictions[:, 1]

    return {
        'accuracy': accuracy,
        'precision_class_0': precision[0],
        'precision_class_1': precision[1],
        'recall_class_0': recall[0],
        'recall_class_1': recall[1],
        'f1_class_0': f1[0],
        'f1_class_1': f1[1],
        'aucpr': auc_precision_recall,
        'train_pickle_file_path': train_pickle_file_path,
        'validation_pickle_file_path': validation_pickle_file_path,
        'test_pickle_file_path': test_pickle_file_path,
        'pos_scores': positive_confidences.tolist(),
        'true_labels': labels.tolist(),
    }


# Load the pickle file
def load_pickle_file(pickle_file_path):
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)
    return data


class MedicalDataset():
    def __init__(self, pickle_file_path):
        all_data = load_pickle_file(pickle_file_path)

        patient_ids = []
        visit_ids = []
        labels = []
        conditions = []
        procedures = []
        drugs = []

        for data in all_data:
            visit_ids.append(data['visit_id'])
            patient_ids.append(data['patient_id'])
            conditions.append(data['conditions'][0])
            procedures.append(data['procedures'][0])
            drugs.append(data['drugs'][0])
            labels.append(data['label'])
        self.data_dict = {
            'visit_id': visit_ids,
            'patient_id': patient_ids,
            'conditions': conditions,
            'procedures': procedures,
            'drugs': drugs,
            'classification_label': labels
        }

    def to_dict(self):
        return self.data_dict


train_dataset = MedicalDataset(train_pickle_file_path)
test_dataset = MedicalDataset(test_pickle_file_path)
validation_dataset = MedicalDataset(validation_pickle_file_path)

# Convert custom datasets to dictionaries
train_dict = train_dataset.to_dict()
test_dict = test_dataset.to_dict()
validation_dict = validation_dataset.to_dict()

# Convert dictionaries to datasets
train_dataset = Dataset.from_dict(train_dict)
test_dataset = Dataset.from_dict(test_dict)
validation_dataset = Dataset.from_dict(validation_dict)

prompt_template = """
Your task is to predict whether a patient is likely to experience readmission based on provided diagnosis, procedures, and drugs information. Analyze the data to assess the likelihood of readmission, considering relevant factors and patterns in the diagnosis, procedures, and drugs data.
Each description is separated by a comma.

**Patient Diagnosis Descriptions:**

{diagnoses}

**Patient Drugs Descriptions:**

{drugs}

**Patient Procedures Descriptions:**

{procedures}

"""


def template_dataset(sample):
    sample["text"] = prompt_template.format(diagnoses=sample["conditions"],
                                            drugs=sample["drugs"],
                                            procedures=sample["procedures"],
                                            eos_token=tokenizer.eos_token)
    return sample


# apply prompt template per sample
train_dataset = train_dataset.map(template_dataset)
print(f'example sample from train:\n {train_dataset[0]}')
validation_dataset = validation_dataset.map(template_dataset)

print(train_dataset[randint(0, len(train_dataset))])

test_dataset = test_dataset.map(template_dataset)
lm_train_dataset = train_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).attention_mask,
        "labels": sample["classification_label"]
    },
    batched=True,
    batch_size=256,
    num_proc=128,
    remove_columns=list(train_dataset.features)
)
lm_validation_dataset = validation_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).attention_mask,
        "labels": sample["classification_label"]
    },
    batched=True,
    batch_size=256,
    num_proc=128,
    remove_columns=list(validation_dataset.features)
)
lm_test_dataset = test_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).attention_mask,
        "labels": sample["classification_label"]
    },
    batched=True,
    batch_size=256,
    num_proc=128,
    remove_columns=list(test_dataset.features)
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy='steps',
    save_strategy='steps',
    eval_steps=1000,
    save_steps=1000,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    auto_find_batch_size=True,
    logging_steps=10,
    learning_rate=2e-5,
    optim="adamw_torch",
    save_total_limit=20,
    # warmup_steps=500,
    # weight_decay=0.002,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="eval_aucpr",
    # gradient_accumulation_steps=8,
    greater_is_better=True,
    dataloader_num_workers=8,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_validation_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

test_results = trainer.evaluate(eval_dataset=lm_test_dataset)

print(f'test_results= {test_results}')
print(f'see outputs in= {OUTPUT_DIR}')
trainer.save_metrics("test", test_results)
