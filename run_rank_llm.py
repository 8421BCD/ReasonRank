import argparse
import os
import sys
import copy
import json
import time
import datetime
from typing import Any, Dict, List, Union, Optional, Sequence
from data import Query, Request, Candidate
import torch
from enum import Enum
from rerank.api_keys import get_openai_api_key
from rerank.rank_gpt import SafeOpenai
from rerank.rank_listwise_os_llm import RankListwiseOSLLM
from rerank.rankllm import PromptMode, RankLLM
from rerank.reranker import Reranker
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
from pyserini.search._base import get_topics, get_qrels
from utils import OutputFormat, get_output_writer, get_qrels_dl22, get_topics_dl22, get_topics_qrels_excluded_ids_for_bright, get_topics_qrels_for_r2med
from dataclasses import dataclass, field
from trec_eval import Eval
from index_and_topics import THE_TOPICS, THE_INDEX, THE_QRELS, BRIGHT_DATASETS, r2med_DATASETS, DOC_MAXLEN
from enum import Enum
from transformers import HfArgumentParser
from tqdm import tqdm
from config import WORKSPACE_DIR, PROJECT_DIR
import random
from datasets import load_dataset

os.environ["PYSERINI_CACHE"] = "../../cache"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class RetrievalMode(Enum):
    DATASET = "dataset"
    CUSTOM = "custom"

    def __str__(self):
        return self.value

class RetrievalMethod(Enum):
    UNSPECIFIED = "unspecified"
    BM25 = "bm25"
    BM25_RM3 = "bm25_rm3"
    SPLADE_P_P_ENSEMBLE_DISTIL = "SPLADE++_EnsembleDistil_ONNX"
    D_BERT_KD_TASB = "distilbert_tas_b"
    OPEN_AI_ADA2 = "openai-ada2"
    REP_LLAMA = "rep-llama"
    CUSTOM_INDEX = "custom_index"
    RESONIR = "reasonir"
    E5_MISTRAL = "e5-mistral-7b-instruct"
    RaDeR = "RaDeR-gte-Qwen2-LLMq_CoT_lexical_hybrid_BM25_Rader"

    def __str__(self):
        return self.value

@dataclass
class Arguments:
    # retrieval arguments
    datasets: List[str] = field(metadata={'help': 'List of test datasets.'})
    output: str = field(metadata={'help': 'Path to output file.'})
    output_format: Optional[str] = field(default='trec', metadata={'help': 'Output format.'})
    retrieval_method: RetrievalMethod = field(default=RetrievalMethod.BM25, metadata={'help': 'Method of retrieval.'})
    retrieval_results_name: str = field(default=None, metadata={'help': 'the filename of custom retrieval results'})
    retrieval_num: int = field(default=100, metadata={'help': 'retrieval number'})
    rerank_topk: int = field(default=None, metadata={'help': 'only need to rerank top-k candidates'})
    threads: int = field(default=30, metadata={'help': 'Number of threads.'})
    batchsize_retrieval: int = field(default=32, metadata={'help': 'batchsize for dense retrieval'})
    reasoning_maxlen: int = field(default=1500, metadata={'help': 'the max length of reasoning chain to generate'})
    remove_query: Optional[bool] = field(default=True, metadata={'help': 'Remove query from output.'})
    save_first_stage_run: Optional[bool] = field(default=True, metadata={'help': 'Save first-stage run.'})
    remove_duplicates: Optional[bool] = field(default=False, metadata={'help': 'Remove duplicates from output.'})
    shuffle_candidates: bool = field(default=False, metadata={'help': 'Whether to shuffle the candidates before reranking.'})
    api_resource: str = field(default='baidu', metadata={'help': ''})
    use_gpt4cot_retrieval: bool = field(default=True, metadata={'help': 'Whether to use the retrieval results of gpt4 cot for BRIGHT datasets (otherwise use the original query).'})

    # llm arguments
    model_path: str = field(default=f'{WORKSPACE_DIR}/llm/rank_vicuna_7b_v1', metadata={'help': 'Path to the model. If `use_azure_ai`, pass your deployment name.'})
    lora_path: str = field(default=None, metadata={'help': 'path of lora'})
    max_lora_rank: int = field(default=32, metadata={'help': 'one parameter of vllm initialization'})
    use_azure_openai: bool = field(default=False, metadata={'help': 'If True, use Azure OpenAI. Requires env var to be set: `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`'})
    context_size: int = field(default=4096, metadata={'help': 'context size used for model.'})
    num_gpus: int = field(default=1, metadata={'help': 'the number of GPUs to use'})
    cache_dir: Optional[str] = field(default='../../cache', metadata={'help': 'Path to cache directory.'})
    llm_dtype: str = field(default='bf16', metadata={'help': 'Data type of llm.'})
    batch_size: int = field(default=32, metadata={'help': 'inference batchsize'})
    num_passes: int = field(default=1, metadata={'help': 'Number of passes to run the model.'})
    window_size: int = field(default=20, metadata={'help': 'Window size for the sliding window approach.'})
    step_size: int = field(default=10, metadata={'help': 'Step size for the sliding window approach.'})
    vllm_batched: bool = field(default=False, metadata={'help': 'Whether to run the model in batches.'})
    prompt_mode: PromptMode = field(default=PromptMode.RANK_GPT, metadata={'required': True, 'choices': list(PromptMode)})
    prompt_info_path: str = field(default=f'{PROJECT_DIR}/listwise_prompt_r1.toml')
    notes: str = field(default='', metadata={'help': 'notes for code running'})



def write_run(output_writer, results, args):
    with output_writer:
        for request in results:
            qid = request.query.qid
            hits = request.candidates
            if args.remove_duplicates:
                seen_docids = set()
                dedup_hits = []
                for hit in hits:
                    if hit.docid.strip() in seen_docids:
                        continue
                    seen_docids.add(hit.docid.strip())
                    dedup_hits.append(hit)
                hits = dedup_hits

            # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
            # We want to remove the query from the results.
            if args.remove_query:
                hits = [hit for hit in hits if hit.docid != qid]

            # write results
            output_writer.write(qid, hits)

def evaluate_results(args, dataset, out_path, qrels, time_cost, current_pass, total_input_tokens, total_output_tokens):
    all_metrics = Eval(out_path, qrels)
    print(f'###################### {dataset} ######################')
    print(all_metrics)
    print(f'time_cost: {time_cost}')
    result = {'model_path': args.model_path,
              'lora_path': args.lora_path,
              'datetime': str(datetime.datetime.now()),
              'retrieval_method': args.retrieval_method,
              'retrieval_num': args.retrieval_num,
              'rerank_topk': args.rerank_topk if args.rerank_topk is not None else 100,
              'current_pass': current_pass,
              'window_size': args.window_size,
              'step_size': args.step_size,
              'shuffle_candidates': args.shuffle_candidates,
              'time_cost': time_cost,
              'total_input_tokens': total_input_tokens,
              'total_output_tokens': total_output_tokens,
              'notes': args.notes,
              **all_metrics}
    os.makedirs('results/', exist_ok=True)
    result_path = f'results/{dataset}.json'
    if os.path.exists(result_path) == False:
        with open(result_path, 'w') as f: 
            json.dump([], f, indent=4)
    with open(result_path, 'r') as f:
        json_data = json.load(f)
        json_data.append(result)
    with open(result_path, 'w') as f: 
        json.dump(json_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(Arguments)
    _args, *_ = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(_args))

    ############################## retrieval for each dataset ##############################
    dataset_qrels = {}
    dataset_topics = {}
    dataset_results = {}
    ### load bright datasets if needed
    if len(set(args.datasets) & set(BRIGHT_DATASETS)) >= 1:
        all_bright_examples = load_dataset('xlangai/bright', 'examples', cache_dir=f'{WORKSPACE_DIR}/data/bright')
    for dataset in args.datasets: 
        if dataset in BRIGHT_DATASETS: # BRIGHT
            topics, qrels, excluded_ids = get_topics_qrels_excluded_ids_for_bright(dataset, all_bright_examples, is_long_doc=False)
            id2doc = {}
            with open(f'{WORKSPACE_DIR}/data/bright/{dataset}/corpus.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    id2doc[data['id']] = data['contents']
        elif dataset in r2med_DATASETS: # R2MED
            topics, qrels = get_topics_qrels_for_r2med(dataset)
            excluded_ids = {qid: [] for qid in list(topics.keys())}
            id2doc = {}
            with open(f'{WORKSPACE_DIR}/data/r2med/{dataset}/corpus.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    id2doc[data['id']] = data['contents']
        else: # TREC and BEIR
            index_path = THE_INDEX[dataset]
            topics_path = THE_TOPICS[dataset]
            qrels_path = THE_QRELS[dataset]

            searcher = LuceneSearcher.from_prebuilt_index(index_path)
            topics = {str(qid): content['title'] for qid, content in get_topics(topics_path).items()} if dataset != 'dl22' else get_topics_dl22() # dl22 is not supported by pyserini 0.20.0
            qrels = {str(qid): {str(docid): int(score) for docid, score in docs.items()} 
                    for qid, docs in get_qrels(qrels_path).items()} if dataset != 'dl22' else get_qrels_dl22()
            excluded_ids = {qid: [] for qid in list(topics.keys())}
        dataset_qrels[dataset] = qrels
        dataset_topics[dataset] = topics
        batch_topic_ids = []
        batch_topics = []
        for topic_id in list(topics.keys()):
            if topic_id in qrels:
                batch_topic_ids.append(topic_id)
                batch_topics.append(topics[topic_id])
        if args.retrieval_results_name is not None:
            first_state_run_path = f'runs/{dataset}/{args.retrieval_results_name}'
        else:
            if args.retrieval_method in ['bm25', 'reasonir']:
                if args.use_gpt4cot_retrieval and dataset in BRIGHT_DATASETS:
                    first_state_run_path = f'runs/{dataset}/{args.retrieval_method}_gpt4cot_top{args.retrieval_num}.txt'
                else:
                    first_state_run_path = f'runs/{dataset}/{args.retrieval_method}_top{args.retrieval_num}.txt'
            else:
                first_state_run_path = f'runs/{dataset}/{args.retrieval_method}_top{args.retrieval_num}.txt'
        if os.path.exists(first_state_run_path):
            print(f'Loading first stage run from {first_state_run_path}.')
            results = []
            with open(first_state_run_path, 'r') as f:
                current_qid = None
                current_ranking = []
                for line in f:
                    if len(line.strip().split()) != 6:  # for stackoverflow dataset, there are several docids containing white space (e.g., pytorch_torch_tensor_functions/Memory Management_2_0.txt), we just filter these docids
                        continue
                    qid, _, docid, _, score, _ = line.strip().split()
                    if qid != current_qid:
                        if current_qid is not None:
                            current_query = Query(qid=current_qid, text=topics[current_qid])
                            results.append(Request(query=current_query, candidates=current_ranking[:args.retrieval_num]))
                        current_ranking = []
                        current_qid = qid
                    current_ranking.append(Candidate(
                                            docid=docid, 
                                            score=float(score), 
                                            doc={'contents': id2doc[docid]} if dataset in BRIGHT_DATASETS + r2med_DATASETS else json.loads(searcher.doc(docid).raw()))
                                            )
                current_query = Query(qid=current_qid, text=topics[current_qid])
                results.append(Request(query=current_query, candidates=current_ranking[:args.retrieval_num]))
        else:
            if args.retrieval_method != 'bm25':
                raise ValueError("the runs of retrieval_method are not found!")
            print(f'First stage run on {dataset} based on bm25...')
            max_excluded_ids_num = max([len(excluded_ids[qid]) for qid in batch_topic_ids])
            _results = searcher.batch_search(batch_topics, batch_topic_ids, k=args.retrieval_num + max_excluded_ids_num, threads=args.threads) # we need to filter the excluded_ids while keeping the retrieved number as retrieval_num
            results = []
            for topic_id in batch_topic_ids:
                candidates = [Candidate(docid=result.docid, score=result.score, doc=json.loads(searcher.doc(result.docid).raw())) 
                                    for result in _results[topic_id] 
                                    if result.docid not in excluded_ids[topic_id]][:args.retrieval_num]
                results.append(Request(query=Query(qid=topic_id, text=topics[topic_id]), candidates=candidates))

            if args.save_first_stage_run:
                output_writer = get_output_writer(first_state_run_path, OutputFormat(args.output_format), 'w',
                                                max_hits=args.retrieval_num, tag=args.retrieval_method, topics=topics, )
                write_run(output_writer, results, args)
        dataset_results[dataset] = results
    ############################# load LLM reranker #############################
    if args.model_path in ['gpt-4o-mini-2024-07-18', 'gpt-4o-mini', 'gpt-4o-2024-08-06', 'gpt-4o-2024-05-13', 'gpt-4o', 'deepseek-chat', 'deepseek-reasoner']:
        agent = SafeOpenai(
            args=args,
            model=args.model_path,
            context_size=args.context_size,
            resource=args.api_resource,
            prompt_mode=args.prompt_mode,
            window_size=args.window_size,
            prompt_info_path=args.prompt_info_path,
            keys=get_openai_api_key(resource=args.api_resource, model_name=args.model_path),
        )
    else:
        print(f"Loading {args.model_path} ...")
        agent = RankListwiseOSLLM(
            args=args,
            model=args.model_path,
            context_size=args.context_size,
            prompt_mode=args.prompt_mode,
            num_gpus=args.num_gpus,
            window_size=args.window_size,
            prompt_info_path=args.prompt_info_path,
            vllm_batched=args.vllm_batched,
        )
    reranker = Reranker(agent)
    ###################################### Reranking ######################################
    for dataset in args.datasets:
        print(f"########################## Reranking on {dataset} ##########################")
        qrels = dataset_qrels[dataset]
        topics = dataset_topics[dataset]
        results = dataset_results[dataset]
        agent.max_passage_length = DOC_MAXLEN[dataset]

        total_time_cost = 0
        for pass_ct in range(args.num_passes):
            print(f"Pass {pass_ct + 1} of {args.num_passes}:")
            # in case of the number of candidate passages < 100 (for sliding windows strategy)
            results_with_100_passages = []
            results_less_100_passages = []
            for result in results:
                if len(result.candidates) < args.retrieval_num:
                    results_less_100_passages.append([result])
                else:
                    results_with_100_passages.append(result)
            results_grouped = [results_with_100_passages[i:i+args.batch_size] for i in range(0, len(results_with_100_passages), args.batch_size)] + results_less_100_passages
            reranked_results = []
            rerank_details = []
            for batch in results_grouped:
                reranked_batch, time_cost, rerank_details_batch = reranker.rerank_batch(
                                            batch,
                                            rank_start=args.step_size * pass_ct,
                                            rank_end=args.retrieval_num if args.rerank_topk is None else args.rerank_topk,
                                            window_size=min(args.window_size, len(batch[0].candidates)),
                                            shuffle_candidates=args.shuffle_candidates,
                                            step=args.step_size,
                                            vllm_batched=args.vllm_batched,
                                        )
                reranked_results.extend(reranked_batch)
                total_time_cost += time_cost
                rerank_details.extend(rerank_details_batch)
            # save results and evaluate
            out_path = os.path.join(f'runs/{dataset}', args.output)
            output_writer = get_output_writer(out_path, OutputFormat(args.output_format), 'w',
                                                max_hits=args.retrieval_num, tag=args.retrieval_method, topics=topics, )
            write_run(output_writer, reranked_results, args)
            total_input_tokens = sum(summary.input_token_count for result in reranked_results for summary in result.ranking_exec_summary)
            total_output_tokens = sum(summary.output_token_count for result in reranked_results for summary in result.ranking_exec_summary)
            evaluate_results(args, dataset, out_path, qrels, time_cost = total_time_cost, current_pass=pass_ct+1, total_input_tokens=total_input_tokens, total_output_tokens=total_output_tokens)
            if args.num_passes > 1:
                results = [
                    Request(copy.deepcopy(r.query), copy.deepcopy(r.candidates))
                    for r in reranked_results
                ]
        print(f"Reranking with {args.num_passes} passes complete!")