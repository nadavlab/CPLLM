import pickle
from random import randint

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve
from transformers import TrainingArguments, AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding

WITH_SUBGROUPS = False

EPOCHS = 0.001
max_length = 4096
output_dir = f"change_me"  # TODO: CHANGE THIS
num_labels = 2
model_id = "meta-llama/Llama-2-13b-hf"
# model_id = "stanford-crfm/BioMedLM"

# MIMIC-IV DATA
# each pickle in the following format: [[[124325,
#   0,
#   ['Mood disorders',
#    'Diabetes mellitus with complications',
#    'Other circulatory disease'],
#   [1, 1, 1]]]
# For more details, see the documentation on GitHub.
train_pickle_file_path = '/sise/home/benshoho/projects/Med-BERT/Fine-Tunning-Tutorials/data/mimic-iv/chronic_kidney_disease_descriptions_train.pickle'  # TODO: CHANGE THIS
validation_pickle_file_path = '/sise/home/benshoho/projects/Med-BERT/Fine-Tunning-Tutorials/data/mimic-iv/chronic_kidney_disease_descriptions_validation.pickle'  # TODO: CHANGE THIS
test_pickle_file_path = '/sise/home/benshoho/projects/Med-BERT/Fine-Tunning-Tutorials/data/mimic-iv/chronic_kidney_disease_descriptions_test.pickle'  # TODO: CHANGE THIS

# set of descriptions for the mimic-iv dataset according the descriptions of the CCS categories.
mimic_iv_description_codes_path = '/sise/home/benshoho/projects/Med-BERT/Pretraining Code/Data Pre-processing Code/mimic_iv_descriptions_set.types'  # TODO: CHANGE THIS
with open(mimic_iv_description_codes_path, 'rb') as file:
    mimic_iv_description_codes = pickle.load(file)


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
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForSequenceClassification.from_pretrained(model_id, quantization_config=bnb_config,
                                                           config=AutoConfig.from_pretrained(model_id,
                                                                                             trust_remote_code=True,
                                                                                             num_labels=num_labels),
                                                           trust_remote_code=True)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

print_trainable_parameters(model)
model = get_peft_model(model, config)
print_trainable_parameters(model)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id


def compute_metrics(p):
    p.predictions = torch.from_numpy(p.predictions)
    p.predictions = torch.softmax(p.predictions, dim=1)

    preds = np.argmax(p.predictions.cpu().numpy(), axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)

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
        'pos_scores': positive_confidences.tolist(),
        'true_labels': labels.tolist(),
    }


import pickle
from datasets import Dataset


class CustomDataset:
    def __init__(self, pickle_file_path):
        with open(pickle_file_path, "rb") as file:
            data = pickle.load(file)

        # Create empty lists to store the dataset
        patient_ids = []
        labels = []
        diagnoses = []
        visits_num = []
        # Extract data from the loaded pickle
        for item in data:
            patient_ids.append(item[0])
            labels.append(item[1])
            diagnoses_list = item[2]
            diagnoses.append(diagnoses_list)
            visits_num.append(item[3])

        self.data_dict = {
            "patient_id": patient_ids,
            "classification_label": labels,
            "diagnoses": diagnoses,
            "visits_num": visits_num
        }

    def to_dict(self):
        return self.data_dict


dataset_type = 'mimic-iv' if 'mimic' in train_pickle_file_path else 'eicu-crd'
print(f'dataset_type={dataset_type}')

train_dataset = CustomDataset(train_pickle_file_path)
test_dataset = CustomDataset(test_pickle_file_path)
validation_dataset = CustomDataset(validation_pickle_file_path)

train_dict = train_dataset.to_dict()
test_dict = test_dataset.to_dict()
validation_dict = validation_dataset.to_dict()

train_dataset = Dataset.from_dict(train_dict)
test_dataset = Dataset.from_dict(test_dict)
validation_dataset = Dataset.from_dict(validation_dict)


def add_mimic_new_tokens_from_diagnosis_strings():
    new_tokens = set()
    for diagnosis_str in mimic_iv_description_codes:
        new_tokens.add(diagnosis_str)

    tokens_to_add = list(new_tokens - set(tokenizer.get_vocab()))
    prev_num_of_tokens = len(tokenizer)
    tokenizer.add_tokens(tokens_to_add)
    tokenizer.add_tokens(['empty_pad'], special_tokens=True)
    new_num_of_tokens = len(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    print(f'first 20 added tokens are= {list(new_tokens)[:20]}')
    print(f'prev tokens number= {prev_num_of_tokens}, new tokens number= {new_num_of_tokens}')


# If you want to use eicu-crd and add tokens to the vocab of pretrained tokenizer, use this:

# def add_eicu_crd_new_tokens_from_diagnosis_strings():
#     new_tokens = set()
#     for diagnosis_str in icd_code_to_icd_string_dict.values():
#         diagnoses_part_of_diagnosis_str = diagnosis_str.split('|')
#         # for words:
#         # for diagnoses_part in diagnoses_part_of_diagnosis_str:
#         #     new_tokens = new_tokens.union(diagnoses_part.split(" "))
#         for diagnoses_part in diagnoses_part_of_diagnosis_str:
#             new_tokens.add(diagnoses_part)
#     # consider to remove special tokens such ( etc..

#     tokens_to_add = list(new_tokens - set(tokenizer.get_vocab()))
#     prev_num_of_tokens = len(tokenizer)
#     tokenizer.add_tokens(tokens_to_add)
#     tokenizer.add_tokens(['empty_pad'], special_tokens=True)
#     new_num_of_tokens = len(tokenizer)
#     model.resize_token_embeddings(len(tokenizer))
#     print(f'first 20 added tokens are= {list(new_tokens)[:20]}')
#     print(f'prev tokens number= {prev_num_of_tokens}, new tokens number= {new_num_of_tokens}')

add_mimic_new_tokens_from_diagnosis_strings()
# add_eicu_crd_new_tokens_from_diagnosis_strings()

prompt_template = """
Your task is to determine whether a patient is likely to have a specific Chronic kidney disease based on their diagnosis descriptions provided below.
Each diagnosis description is separated by a comma.

**Patient Diagnosis Descriptions:**

{diagnoses}
"""


def template_dataset(sample):
    sample["text"] = prompt_template.format(diagnoses=sample["diagnoses"],
                                            eos_token=tokenizer.eos_token)
    return sample


# apply prompt template per sample
train_dataset = train_dataset.map(template_dataset)
print(f'example sample from train:\n {train_dataset[0]}')
validation_dataset = validation_dataset.map(template_dataset)

print(train_dataset[randint(0, len(train_dataset))]["diagnoses"])

# apply prompt template per sample
test_dataset = test_dataset.map(template_dataset)
lm_train_dataset = train_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=max_length).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=max_length).attention_mask,
        "labels": sample["classification_label"]
    },
    batched=True,
    batch_size=64,
    remove_columns=list(train_dataset.features)
)
lm_validation_dataset = validation_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=max_length).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=max_length).attention_mask,
        "labels": sample["classification_label"]
    },
    batched=True,
    batch_size=64,
    remove_columns=list(validation_dataset.features)
)
lm_test_dataset = test_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=max_length).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=max_length).attention_mask,
        "labels": sample["classification_label"]
    },
    batched=True,
    batch_size=64,
    remove_columns=list(test_dataset.features)
)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

labels = lm_train_dataset['labels']

# Calculate class weights for an imbalanced dataset
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f'class_weights= {class_weights}')

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

from transformers import Trainer

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy='steps',
    save_strategy='steps',
    eval_steps=50,
    save_steps=50,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    auto_find_batch_size=True,
    logging_steps=30,
    learning_rate=2e-5,
    optim="adamw_torch",
    save_total_limit=20,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="eval_aucpr",
    greater_is_better=True,
    dataloader_num_workers=8,
)

model.config.use_cache = False
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_validation_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()
trainer.save_model(output_dir)

test_results = trainer.evaluate(eval_dataset=lm_test_dataset)

print(f'see outputs in= {output_dir}')
trainer.save_metrics("test", test_results)
