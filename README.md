# CPLLM: Clinical Prediction with Large Language Models

This repository contains the code and resources for the paper titled "CPLLM: Clinical Prediction with Large Language Models."

If you use CPLLM or find this repository useful for your research or work, please cite us using the following citation:
```
@article{shoham2024cpllm,
  title={Cpllm: Clinical prediction with large language models},
  author={Shoham, Ofir Ben and Rappoport, Nadav},
  journal={PLOS Digital Health},
  volume={3},
  number={12},
  pages={e0000680},
  year={2024}
}
```

## Getting Started

To get started with CPLLM, follow these steps:

### 1. Install Conda Environment

Use the provided `environment.yml` file to create a Conda environment with the necessary dependencies. Run the following command to create the environment:

```bash
conda env create -f environment.yml
conda activate cpllm-env
```

### 2. Data Extraction
You can use the provided Jupyter notebooks to create the data required for fine-tuning the model. We have two notebooks for data extraction:

2.1) Data Extraction for Next Diagnosis Prediction:
Use the `medbert-fine-tuning-data-extraction-eicu_crd.ipynb` notebook to extract data for next diagnosis prediction.

2.2) Data Extraction for Next Visit Diagnosis Prediction:
Use the `medbert-fine-tuning-data-extraction-mimic-iv.ipynb` notebook to extract data for next visit diagnosis prediction.

2.3) Data Extraction for Readmission Prediction

`pip install pyhealth`

Then, use the script available at https://github.com/nadavlab/CPLLM/blob/main/readmission-data-extraction.py to utilize pyhealth for readmission data extraction.

### 3. Fine-Tuning
After extracting the required data, you can fine-tune the CPLLM model. Make sure to modify the configuration variables in the `cpllm.py` code to suit your specific use case.

Run the training of CPLLM:
`python cpllm_disease_prediction.py` for disease preidction. And `python cpllm_readmission_prediction.py` for readmission prediction.


