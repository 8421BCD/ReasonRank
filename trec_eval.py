import json
import os
import re
import subprocess
import sys
import platform
import pandas as pd
import tempfile
import pytrec_eval
from pyserini.search import get_qrels_file
from pyserini.util import download_evaluation_script
import numpy as np


def Eval(run_trec_file, qrel_trec_file, result_path=None, rel_threshold=None):
    # process run trec file
    if isinstance(run_trec_file, dict):
        runs = run_trec_file
    else:
        with open(run_trec_file, 'r', encoding='utf-8') as f:
            run_data = f.readlines()
        runs = {}
        for line in run_data:
            line = line.split(" ")
            sample_id = line[0]
            doc_id = line[2]
            score = float(line[4])
            if sample_id not in runs:
                runs[sample_id] = {}
            runs[sample_id][doc_id] = score

    # process qrel trec file
    if isinstance(qrel_trec_file, dict):
        qrels_ndcg = qrel_trec_file
    else:
        if qrel_trec_file.endswith('.json'): # json file
            with open(qrel_trec_file, 'r') as f:
                qrels_ndcg = json.load(f)
            for qid, docs in qrels_ndcg.items():
                qrels_ndcg[qid] = {key: int(val) for key, val in docs.items()}
        else:   # trec format
            with open(qrel_trec_file, 'r') as f:
                qrel_data = f.readlines()
            # qrels = {}
            qrels_ndcg = {}
            for line in qrel_data:
                line = line.strip().split("\t")
                if len(line) == 4: # qid _ did rel
                    query = line[0]
                    doc_id = line[2]
                    rel = int(line[3])
                elif len(line) == 3: # qid did rel
                    query = line[0]
                    doc_id = line[1]
                    rel = int(line[2])
                # if query not in qrels:
                #     qrels[query] = {}
                if query not in qrels_ndcg:
                    qrels_ndcg[query] = {}

                # for NDCG
                qrels_ndcg[query][doc_id] = rel

                # for MAP, MRR, Recall
                # if rel >= rel_threshold:
                #     rel = 1
                # else:
                #     rel = 0
                # qrels[query][doc_id] = rel


    # pytrec_eval eval
    # evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    # res = evaluator.evaluate(runs)
    # map_list = [v['map'] for v in res.values()]
    # mrr_list = [v['recip_rank'] for v in res.values()]
    # recall_5_list = [v['recall_5'] for v in res.values()]
    # recall_10_list = [v['recall_10'] for v in res.values()]
    # recall_20_list = [v['recall_20'] for v in res.values()]
    # recall_100_list = [v['recall_100'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {'ndcg_cut.1,5,10,20,50'})
    res = evaluator.evaluate(runs)

    ndcg_1_list = [v['ndcg_cut_1'] for v in res.values()]
    ndcg_5_list = [v['ndcg_cut_5'] for v in res.values()]
    ndcg_10_list = [v['ndcg_cut_10'] for v in res.values()]
    ndcg_20_list = [v['ndcg_cut_20'] for v in res.values()]
    ndcg_50_list = [v['ndcg_cut_50'] for v in res.values()]
    res = {
            # "MAP": np.average(map_list),
            # "MRR": np.average(mrr_list),
            # "Recall@5": np.average(recall_5_list),
            # "Recall@10": np.average(recall_10_list),
            # "Recall@20": np.average(recall_20_list),
            # "Recall@100": np.average(recall_100_list),
            "NDCG@1": '{:.2f}'.format(np.average(ndcg_1_list) * 100), 
            "NDCG@5": '{:.2f}'.format(np.average(ndcg_5_list) * 100), 
            "NDCG@10": '{:.2f}'.format(np.average(ndcg_10_list) * 100), 
            # "NDCG@20": '{:.2f}'.format(np.average(ndcg_20_list) * 100), 
            # "NDCG@50": '{:.2f}'.format(np.average(ndcg_50_list) * 100), 
        }
    return res


class EvalFunction:
    @staticmethod
    def trunc(qrels, run):
        qrels = get_qrels_file(qrels)
        run = pd.read_csv(run, delim_whitespace=True, header=None)
        qrels = pd.read_csv(qrels, delim_whitespace=True, header=None)
        run[0] = run[0].astype(str)
        qrels[0] = qrels[0].astype(str)

        qrels = qrels[qrels[0].isin(run[0])]
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels.to_csv(temp_file, sep='\t', header=None, index=None)
        return temp_file

    @staticmethod
    def eval(args, trunc=True):
        script_path = download_evaluation_script('trec_eval')
        cmd_prefix = ['java', '-jar', script_path]
        # args = sys.argv

        # Option to discard non-judged hits in run file
        judged_docs_only = ''
        judged_result = []
        cutoffs = []

        if '-remove-unjudged' in args:
            judged_docs_only = args.pop(args.index('-remove-unjudged'))

        if any([i.startswith('judged.') for i in args]):
            # Find what position the arg is in.
            idx = [i.startswith('judged.') for i in args].index(True)
            cutoffs = args.pop(idx)
            cutoffs = list(map(int, cutoffs[7:].split(',')))
            # Get rid of the '-m' before the 'judged.xxx' option
            args.pop(idx - 1)

        temp_file = ''

        if len(args) > 1:
            if trunc:
                args[-2] = EvalFunction.trunc(args[-2], args[-1])
                # print('Trunc', args[-2])

            if not os.path.exists(args[-2]):
                args[-2] = get_qrels_file(args[-2])
            if os.path.exists(args[-1]):
                # Convert run to trec if it's on msmarco
                with open(args[-1]) as f:
                    first_line = f.readline()
                if 'Q0' not in first_line:
                    temp_file = tempfile.NamedTemporaryFile(delete=False).name
                    print('msmarco run detected. Converting to trec...')
                    run = pd.read_csv(args[-1], delim_whitespace=True, header=None,
                                      names=['query_id', 'doc_id', 'rank'])
                    run['score'] = 1 / run['rank']
                    run.insert(1, 'Q0', 'Q0')
                    run['name'] = 'TEMPRUN'
                    run.to_csv(temp_file, sep='\t', header=None, index=None)
                    args[-1] = temp_file

            run = pd.read_csv(args[-1], delim_whitespace=True, header=None)
            qrels = pd.read_csv(args[-2], delim_whitespace=True, header=None)

            # cast doc_id column as string
            run[0] = run[0].astype(str)
            qrels[0] = qrels[0].astype(str)

            # Discard non-judged hits

            if judged_docs_only:
                if not temp_file:
                    temp_file = tempfile.NamedTemporaryFile(delete=False).name
                judged_indexes = pd.merge(run[[0, 2]].reset_index(), qrels[[0, 2]], on=[0, 2])['index']
                run = run.loc[judged_indexes]
                run.to_csv(temp_file, sep='\t', header=None, index=None)
                args[-1] = temp_file
            # Measure judged@cutoffs
            for cutoff in cutoffs:
                run_cutoff = run.groupby(0).head(cutoff)
                judged = len(pd.merge(run_cutoff[[0, 2]], qrels[[0, 2]], on=[0, 2])) / len(run_cutoff)
                metric_name = f'judged_{cutoff}'
                judged_result.append(f'{metric_name:22}\tall\t{judged:.4f}')
            cmd = cmd_prefix + args[1:]
        else:
            cmd = cmd_prefix

        # print(f'Running command: {cmd}')
        shell = platform.system() == "Windows"
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=shell)
        stdout, stderr = process.communicate()
        if stderr:
            print(stderr.decode("utf-8"))

        # print('Results:')
        # print(stdout.decode("utf-8").rstrip())

        # for judged in judged_result:
        #     print(judged)

        if temp_file:
            os.remove(temp_file)
        metric_name, _, metrics_value = stdout.decode("utf-8").rstrip().split('\t')
        return "{:.2f}".format(float(metrics_value) * 100)
