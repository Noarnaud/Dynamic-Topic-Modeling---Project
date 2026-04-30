# Dynamic-Topic-Modeling-NLP-project

# Structure Project

```
├── README.md
├── Notebooks : Notebooks for analysis
│
├── nlp-lab-topic-main #dossiers avec les scripts des TDs, pas necessaire dans ce projet
├── data
│      ├── legislatives_73 
│      ├── legislatives_78 
│      ├── legislatives_81 
│      ├── legislatives_88 
│      ├── legislativearchelect_search.csv: metadata of the documents 
├── src/
│      ├── __init__.py
│      ├── config.py          Central configuration (paths, hyperparams)
│      ├── data_loader.py     Load and clean data from Archelec
│      ├── lda_model.py       LDA Baseline
│      ├── bert_model.py      Bert model
│      ├── coherence.py       quantitative evaluation of models
│      └── visualization.py   functions to plot visualizations
│
├── main.py                
├── outputs/         
│
├── .env 
├── .gitignore
├── requirements.txt
└── 
```

# How to run the project 
## On SSPCloud : 

## Create a virtuel environment :

`python -m venv venv_NLP`

## Activate the environment

`source venv_NLP/bin/activate` sur SSP cloud

## Install the require libraries

`pip install -r requirements.txt`


## Run the code: 

  python main.py                           # Pipeline complète (4 années, 10 topics)
  python main.py --years 81                # Une seule année
  python main.py --n-topics 15 --no-cache  # 15 topics, sans cache
  python main.py --skip-lda                # Seulement BERTopic
  python main.py --skip-bert               # Seulement LDA
