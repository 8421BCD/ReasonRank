# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import rbo
import toml
import pytrec_eval
from typing import List

pattern = toml.load('{YOUR_PROJECT_DIR}/listwise_prompt_r1.toml')['pattern']

def clean_response(response: str):
    new_response = ""
    for c in response:
        if not c.isdigit():
            new_response += " "
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response

def compute_metric(pred_docids, relevant_docids):
    qid = '1' # dummy qid '1'
    pred_results = {
        qid: {docid: -(idx+1) for idx, docid in enumerate(pred_docids)}  
    }
    gt = {
        qid: {docid: 1 for docid in relevant_docids}  # dummy qid '1'
    }
    evaluator = pytrec_eval.RelevanceEvaluator(gt, {'ndcg_cut.10', 'recall.10'})
    res = evaluator.evaluate(pred_results)
    return res[qid]['ndcg_cut_10'], res[qid]['recall_10']


def compute_score(predict_str: str, ground_truth: str, initial_list: List[str], final_list: List[str], relevant_docids: List[int]) -> float:
    match = re.search(pattern, predict_str, re.DOTALL)
    if match:
        pred = match.group(1)
        _pred_id_list = clean_response(pred).split()
        label_id_list = clean_response(ground_truth).split()
        if len(_pred_id_list) == len(label_id_list) and len(set(_pred_id_list)) == len(set(label_id_list)):  # ensure the number of ids 
            try:
                pred_idxs = [int(x) - 1 for x in _pred_id_list]
            except ValueError: # unexcepted output
                return 0

            pred_docids = [initial_list[idx] for idx in pred_idxs if idx < len(initial_list)]
            ndcg10, recall10 = compute_metric(pred_docids, relevant_docids)
            rbo_score = rbo.RankingSimilarity(_pred_id_list, label_id_list).rbo()
            reward_score = ndcg10 + 0.2 * recall10 + 0.1 * rbo_score
            return reward_score
        else:
            return 0
    else:
        return -1

    # without reasoning
    # _pred_id_list = clean_response(predict_str).split()
    # label_id_list = clean_response(ground_truth).split()
    # if len(_pred_id_list) == len(label_id_list) and len(set(_pred_id_list)) == len(set(label_id_list)):
    # # pred_id_list = []
    # # for _id in _pred_id_list:
    # #     if _id not in pred_id_list:
    # #         pred_id_list.append(_id)
    #     ranksim = rbo.RankingSimilarity(_pred_id_list, label_id_list).rbo() # to do
    #     return ranksim
    # else:
    #     return 0
