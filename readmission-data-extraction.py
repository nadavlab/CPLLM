from pyhealth.medcode import InnerMap
from tqdm import tqdm
from pyhealth.datasets import eICUDataset  # pip install pyhealth is required.
import pickle
from pyhealth.datasets import split_by_patient

OUTPUT_DIR = '/sise/nadav-group/nadavrap-group/ofir/pyhealth_data/readmission'

# For MIMIC: Please refer to the PyHealth documentation.
# If you need assistance, feel free to open an issue at https://github.com/nadavlab/CPLLM/issues.
eicu_crd_ds = eICUDataset(
    root="/home/benshoho/storage/eicu_crd/physionet.org/files/eicu-crd/2.0",
    tables=["diagnosis", "medication", "physicalExam"],
    code_mapping={"ICD9CM": "CCSCM", "ICD10CM": "CCSCM", "ICD9PROC": "CCSPROC", "ICD10PROC": "CCSPROC", "NDC": "ATC"},
)

print(eicu_crd_ds.stat())

from pyhealth.tasks import mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn, \
    readmission_prediction_eicu_fn, readmission_prediction_eicu_fn2, mortality_prediction_eicu_fn

eicu_crd_task_ds = eicu_crd_ds.set_task(task_fn=readmission_prediction_eicu_fn)
print("eicu_crd_task_ds:")
print(eicu_crd_task_ds.stat())


def convert(code, code_to_descriptions_dict, added_dict):
    if code in added_dict:
        return added_dict[code]
    try:
        return code_to_descriptions_dict.lookup(code)
    except:
        return code


def from_codes_to_descriptions(task_dataset_with_codes):
    task_dataset_with_descriptions = []
    durgs_dict = InnerMap.load("ATC", refresh_cache=True)
    procedors_ccs_dict = InnerMap.load("CCSPROC", refresh_cache=True)
    icd_ccs_dict = InnerMap.load("CCSCM", refresh_cache=True)
    added_dict = {'V03AG05': 'sodium phosphate', 'L04AA52': 'ofatumumab'}

    for sample in tqdm(task_dataset_with_codes):
        new_sample = sample
        for name, code_to_descriptions_dict in {'conditions': icd_ccs_dict, 'procedures': procedors_ccs_dict,
                                                'drugs': durgs_dict}.items():
            try:
                new_sample[name] = [[convert(code, code_to_descriptions_dict, added_dict) for code in sublist] for
                                    sublist in sample[name]]
            except Exception as e:
                print(f'name={name}, e={e}')

        # Move the append outside the inner loop
        task_dataset_with_descriptions.append(new_sample)

    return task_dataset_with_descriptions


def write_dataset_to_pickle(ds, ds_name, output_dir):
    descriptions_ds = from_codes_to_descriptions(ds)
    print(ds_name, " with descriptions: ", len(descriptions_ds))
    with open(f'{output_dir}/eicu_crd_readmission_prediction_with_descriptions_{ds_name}.pickle', 'wb') as fp2:
        pickle.dump(descriptions_ds, fp2, protocol=pickle.HIGHEST_PROTOCOL)


# data split
train_dataset, val_dataset, test_dataset = split_by_patient(eicu_crd_task_ds, [0.8, 0.1, 0.1])
with open(f'{OUTPUT_DIR}/eicu_crd_readmission_prediction_train.pickle', 'wb') as fp1:
    pickle.dump(train_dataset, fp1, pickle.HIGHEST_PROTOCOL)
with open(f'{OUTPUT_DIR}/eicu_crd_readmission_prediction_validation.pickle', 'wb') as fp2:
    pickle.dump(val_dataset, fp2, pickle.HIGHEST_PROTOCOL)
with open(f'{OUTPUT_DIR}/eicu_crd_readmission_prediction_test.pickle', 'wb') as fp3:
    pickle.dump(test_dataset, fp3, pickle.HIGHEST_PROTOCOL)

write_dataset_to_pickle(train_dataset, 'train', OUTPUT_DIR)
write_dataset_to_pickle(val_dataset, 'validation', OUTPUT_DIR)
write_dataset_to_pickle(test_dataset, 'test', OUTPUT_DIR)

print('done!')
