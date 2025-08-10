import numpy as np
import json
from config import WORKSPACE_DIR

THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    "dl21": "msmarco-v2-passage",
    "dl22": "msmarco-v2-passage",
    "dl23": "msmarco-v2-passage",
    'msmarco': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid-flat',
    'arguana': 'beir-v1.0.0-arguana-flat',
    'touche': 'beir-v1.0.0-webis-touche2020-flat',
    'news': 'beir-v1.0.0-trec-news-flat',
    'scifact': 'beir-v1.0.0-scifact-flat',
    'fiqa': 'beir-v1.0.0-fiqa-flat',
    'scidocs': 'beir-v1.0.0-scidocs-flat',
    'nfc': 'beir-v1.0.0-nfcorpus-flat',
    'quora': 'beir-v1.0.0-quora-flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'climate-fever': 'beir-v1.0.0-climate-fever-flat',
    'robust04': 'beir-v1.0.0-robust04-flat',
    'signal': 'beir-v1.0.0-signal1m-flat',
    'hotpotqa': 'beir-v1.0.0-hotpotqa-flat',
    'nq': 'beir-v1.0.0-nq-flat',
    'nfcorpus': 'beir-v1.0.0-nfcorpus-flat',
    'bioasq': 'beir-v1.0.0-bioasq-flat',


    'mrtydi-ar': 'mrtydi-v1.1-arabic',
    'mrtydi-bn': 'mrtydi-v1.1-bengali',
    'mrtydi-fi': 'mrtydi-v1.1-finnish',
    'mrtydi-id': 'mrtydi-v1.1-indonesian',
    'mrtydi-ja': 'mrtydi-v1.1-japanese',
    'mrtydi-ko': 'mrtydi-v1.1-korean',
    'mrtydi-ru': 'mrtydi-v1.1-russian',
    'mrtydi-sw': 'mrtydi-v1.1-swahili',
    'mrtydi-te': 'mrtydi-v1.1-telugu',
    'mrtydi-th': 'mrtydi-v1.1-thai',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20',
    "dl21": "dl21",
    "dl22": "dl22",
    'msmarco': 'msmarco-passage-dev-subset',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',
    'fever': 'beir-v1.0.0-fever-test',
    'climate-fever': 'beir-v1.0.0-climate-fever-test',
    'hotpotqa': 'beir-v1.0.0-hotpotqa-test',
    'nfcorpus': 'beir-v1.0.0-nfcorpus-test',
    'nq': 'beir-v1.0.0-nq-test',
    'bioasq': 'beir-v1.0.0-bioasq-test',



    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',
}

THE_QRELS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    "dl21": "dl21-passage",
    "dl22": "dl22-passage",
    'msmarco': 'msmarco-passage-dev-subset',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',
    'fever': 'beir-v1.0.0-fever-test',
    'climate-fever': 'beir-v1.0.0-climate-fever-test',
    'hotpotqa': 'beir-v1.0.0-hotpotqa-test',
    'nfcorpus': 'beir-v1.0.0-nfcorpus-test',
    'nq': 'beir-v1.0.0-nq-test',
    'bioasq': 'beir-v1.0.0-bioasq-test',



    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',
}

BRIGHT_DATASETS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
]

r2med_DATASETS = [
    'r2med_Medical-Sciences', 
    'r2med_Biology', 
    'r2med_MedQA-Diag', 
    'r2med_PMC-Clinical', 
    'r2med_Bioinformatics', 
    'r2med_MedXpertQA-Exam', 
    'r2med_PMC-Treatment', 
    'r2med_IIYi-Clinical'
]

# DATA_DIR = f'{WORKSPACE_DIR}/data'

# THE_QRELS = {
#     'dl19': f'{DATA_DIR}/ms_marco/dl19/qrels.json',
#     'dl20': f'{DATA_DIR}/ms_marco/dl20/qrels.json',
#     'msmarco': f'{DATA_DIR}/ms_marco/passage_ranking/qrels/qrels.dev.tsv',
#     'fever': f'{DATA_DIR}/beir/fever/qrels/test.tsv',
#     'climate-fever': f'{DATA_DIR}/beir/climate-fever/qrels/test.tsv',
#     'fiqa': f'{DATA_DIR}/beir/fiqa/qrels/test.tsv',
#     'hotpotqa': f'{DATA_DIR}/beir/hotpotqa/qrels/test.tsv',
#     'nfcorpus': f'{DATA_DIR}/beir/nfcorpus/qrels/test.tsv',
#     'nq': f'{DATA_DIR}/beir/nq/qrels/test.tsv',
#     'quora': f'{DATA_DIR}/beir/quora/qrels/test.tsv',
#     'scifact': f'{DATA_DIR}/beir/scifact/qrels/test.tsv',
#     'dbpedia': f'{DATA_DIR}/beir/dbpedia/qrels/test.tsv',
#     'covid': f'{DATA_DIR}/beir/covid/qrels/test.tsv',
#     'arguana': f'{DATA_DIR}/beir/arguana/qrels/test.tsv',
#     'touche': f'{DATA_DIR}/beir/touche/qrels/test.tsv',
#     'scidocs': f'{DATA_DIR}/beir/scidocs/qrels/test.tsv',
#     'news': f'{DATA_DIR}/beir/news/qrels/test.tsv',
#     'robust04': f'{DATA_DIR}/beir/robust04/qrels/test.tsv',
#     'signal': f'{DATA_DIR}/beir/signal/qrels/test.tsv',
# }

# calculate by np.percentile(, 90)
# DOC_MAXLEN = {
#     'msmarco': 100,
#     'dl20': 100,
#     'dl19': 100,
#     'scifact': 300,
#     'fiqa': 150, # 277
#     'nfcorpus': 300,
#     'fever': 220,
#     'hotpotqa': 90,
#     'quora': 100
# }

# not used
QUERY_MAXLEN = {
    'msmarco': 64,
    'dl20': 64,
    'dl19': 64,
    'scifact': 64,
    'fiqa': 64,
    'nfcorpus': 64,
    'fever': 64,
    'climate-fever': 64,
    'hotpotqa': 64,
    'quora': 64,
    'dbpedia':64,
    'nq': 64,
    'covid': 64,
    'arguana': 64,
    'touche': 64,
    'scidocs': 64,
    'news': 64,
    'robust04': 64,
    'signal': 64,
}

DOC_MAXLEN = {
    # trec
    'msmarco': 100,
    'dl19': 100,
    'dl20': 100,
    'dl21': 100,
    'dl22': 100,
    # beir
    'scifact': 100,
    'fiqa': 100,
    'nfcorpus': 100,
    'fever': 100,
    'climate-fever': 100,
    'hotpotqa': 100,
    'quora': 100,
    'dbpedia':100,
    'nq': 100,
    'covid': 100,
    'arguana': 100,
    'touche': 100,
    'scidocs': 100,
    'news': 100,
    'robust04': 100,
    'signal': 100,

    # bright benchmark
    "biology": 512,
    "earth_science": 512,
    "economics": 512,
    "psychology": 512,
    "robotics": 512,
    "stackoverflow": 512,
    "sustainable_living": 512,
    "leetcode": 512,
    "pony": 512,
    "aops": 512,
    "theoremqa_questions": 512,
    "theoremqa_theorems": 512,
    
    # r2med benchmark
    "r2med_Medical-Sciences": 512,
    "r2med_Biology": 512,
    "r2med_MedQA-Diag": 512,
    "r2med_PMC-Clinical": 512,
    "r2med_Bioinformatics": 512,
    "r2med_MedXpertQA-Exam": 512,
    "r2med_PMC-Treatment": 512,
    "r2med_IIYi-Clinical": 512,

    # for math data of reasonrank 
    "math-qa": 512,
    "math-theorem": 512,
}




if __name__ == '__main__':
    # datasets = ['scifact', 'fiqa', 'nfcorpus', 'fever', 'hotpotqa', 'quora']
    datasets = ['nq']
    # a = {'1': '12321', '2': 'dfdf'}
    # print(list(a.values()))
    # exit()
    for dataset in datasets:
        dataset_id_doc_path = f'../../data/beir/{dataset}/id_doc.json'
        with open(dataset_id_doc_path, 'r') as f: 
            id_doc = json.load(f)
        len_list = [len(doc.split(' ')) for id, doc in id_doc.items()]
        print(dataset, np.percentile(len_list, 90))
        queries_path = f'../../data/beir/{dataset}/queries.jsonl'
        len_list = []
        with open(queries_path, 'r') as f: 
            for line in f: 
                data = json.loads(line)
                len_list.append(len(data['text'].split(' ')))
        print(dataset, np.percentile(len_list, 95))