from datetime import datetime
from pathlib import Path
from typing import List
import json
from tqdm import tqdm

from data import DataWriter, Request, Result
from rerank.rankllm import RankLLM


class Reranker:
    def __init__(self, agent: RankLLM) -> None:
        self._agent = agent

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        window_size: int = 20,
        step: int = 10,
        shuffle_candidates: bool = False,
        logging: bool = False,
        vllm_batched: bool = False,
        populate_exec_summary: bool = True,
        num_beams: int = None,
    ) -> List[Result]:
        """
        Reranks a list of requests using the RankLLM agent.

        This function applies a sliding window algorithm to rerank the results.
        Each window of results is processed by the RankLLM agent to obtain a new ranking.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            window_size (int, optional): The size of each sliding window. Defaults to 20.
            step (int, optional): The step size for moving the window. Defaults to 10.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            vllm_batched (bool, optional): Whether to use VLLM batched processing. Defaults to False.
            generate_training_data (bool, optional): whether to generate training data or just inference
            populate_exec_summary (bool, optional): Whether to populate the exec summary. Defaults to False.

        Returns:
            List[Result]: A list containing the reranked candidates.
        """
        if vllm_batched:
            for i in range(1, len(requests)):
                assert len(requests[0].candidates) == len(requests[i].candidates), "Batched requests must have the same number of candidates"
            print('using vllm_batched...')
            initial_passage_list_batch = [[candidate.docid for candidate in request.candidates] for request in requests]
            rerank_results, total_time_cost, rerank_details_batch = self._agent.sliding_windows_batched(
                requests,
                rank_start=max(rank_start, 0),
                rank_end=min(rank_end, len(requests[0].candidates)),  # TODO: Fails arbitrary hit sizes
                window_size=min(window_size, len(requests[0].candidates)),
                step=step,
                shuffle_candidates=shuffle_candidates,
                logging=logging,
            )
            return rerank_results, total_time_cost, rerank_details_batch

        rerank_results = []
        rerank_details_batch = []
        total_time_cost = 0
        for request in tqdm(requests):
        # for request in requests:
            initial_passage_list = [candidate.docid for candidate in request.candidates]
            rerank_result, time_cost, rerank_details = self._agent.sliding_windows(
                                                    request,
                                                    rank_start=max(rank_start, 0),
                                                    rank_end=min(rank_end, len(request.candidates)),
                                                    window_size=min(window_size, len(request.candidates)),
                                                    step=step,
                                                    shuffle_candidates=shuffle_candidates,
                                                    logging=logging,
                                                    num_beams=num_beams,
                                                    )
            rerank_results.append(rerank_result)
            rerank_details_batch.append(rerank_details)
            total_time_cost += time_cost
        return rerank_results, total_time_cost, rerank_details_batch

