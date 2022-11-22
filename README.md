# GPTNERMED

## About
GPTNERMED is a novel open synthesized dataset and neural named-entity-recognition (NER) model for German texts in medical natural language processing (NLP).

Key features:
 - Supported labels: *Medikation*, *Dosis*, *Diagnose*
 - Open silver-standard German medical dataset: **245107 tokens** with annotations for Dosis (**#7547**), Medikation (**#9868**) and Diagnose (**#5996**)
 - Synthesized dataset based on [**GPT NeoX**](https://github.com/EleutherAI/gpt-neox)
 - **Transfer-learning** for NER parsing using **gbert-large**, **GottBERT-base** or **German-MedBERT**
 - **Open, public access** to models

**Online Demo**: A demo page is available: [Demo](https://gptnermed.misit-augsburg.de/)

See our **[paper](https://arxiv.org/pdf/2208.14493.pdf)** at [https://arxiv.org/pdf/2208.14493.pdf](https://arxiv.org/pdf/2208.14493.pdf).

NER demonstration:  
<kbd><img src="./demo.png" alt="NER example demo" width="600"/></kbd>

## Models
The pretrained models can be retrieved from the following URLs:
- gbert-based: [model link](https://myweb.rz.uni-augsburg.de/~freijoha/GPTNERMED/GPTNERMED_gbert.zip)
- GottBERT-based: [model link](https://myweb.rz.uni-augsburg.de/~freijoha/GPTNERMED/GPTNERMED_GottBERT.zip)
- German-MedBERT-based: [model link](https://myweb.rz.uni-augsburg.de/~freijoha/GPTNERMED/GPTNERMED_GermanMedBERT.zip)

## Scores
Note: Metric scores are evaluated by character-wise classification.

**Out of Distribution Dataset** (provided in `OoD-dataset_GoldStandard.jsonl`):  
| **Model**          | Metric | **Drug = Medikation** |
|--------------------|--------|-----------------------|
| **gbert-large**    | Pr     | 0.707                 |
|                    | Re     | **0.979**             |
|                    | F1     | 0.821                 |
| **GottBERT-base**  | Pr     | **0.800**             |
|                    | Re     | 0.899                 |
|                    | F1     | **0.847**             |
| **German-MedBERT** | Pr     | 0.727                 |
|                    | Re     | 0.818                 |
|                    | F1     | 0.770                 |

**Test Set**:  
| **Model**          | Metric | **Medikation** | **Diagnose** | **Dosis** | **Total** |
|--------------------|--------|----------------|--------------|-----------|-----------|
| **gbert-large**    | Pr     | 0.870          | 0.870        | 0.883     | 0.918     |
|                    | Re     | **0.936**      | **0.895**    | **0.921** | **0.919** |
|                    | F1     | **0.949**      | **0.882**    | **0.901** | **0.918** |
| **GottBERT-base**  | Pr     | 0.979          | 0.896        | **0.887** | **0.936** |
|                    | Re     | 0.910          | 0.844        | 0.907     | 0.886     |
|                    | F1     | 0.943          | 0.870        | 0.897     | 0.910     |
| **German-MedBERT** | Pr     | **0.980**      | **0.910**    | 0.829     | 0.932     |
|                    | Re     | 0.905          | 0.730        | 0.890     | 0.842     |
|                    | F1     | 0.941          | 0.810        | 0.858     | 0.883     |

## Setup and Usage
The models are based on SpaCy. The sample code is written in Python.

```bash
model_link="https://myweb.rz.uni-augsburg.de/~freijoha/GPTNERMED/GPTNERMED_gbert.zip"

# [Optional] Create env
python3 -m venv env
source ./env/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt

# Download & extract model
wget -O model.zip "$model_link"
unzip model.zip -d "model"

# Run script
python3 GPTNERMED.py
```
