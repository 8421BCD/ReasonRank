import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


import copy
import random
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Tuple, Union
from ftfy import fix_text
from tqdm import tqdm
from data import RankingExecInfo, Request, Result
import time
import json
import toml

class PromptMode(Enum):
    UNSPECIFIED = "unspecified"
    RANK_GPT = "rank_GPT"
    RANK_GPT_reasoning = "rank_GPT_reasoning"
    RANK_GPT_rankk = "rank_GPT_rankk"
    RANK_GPT_qwen3 = "rank_GPT_qwen3"
    RANK_GPT_APEER = "rank_GPT_APEER"
    LRL = "LRL"

    def __str__(self):
        return self.value

class RankLLM(ABC):
    def __init__(
        self,
        model: str,
        context_size: int,
        prompt_mode: PromptMode,
        prompt_info_path: str,
        num_few_shot_examples: int,
    ) -> None:
        self._model = model
        self._context_size = context_size
        self._num_few_shot_examples = num_few_shot_examples
        self.prompt_mode = prompt_mode
        self.prompt_info = toml.load(prompt_info_path) if prompt_info_path is not None else None
        self.processed_queries = 0

    def max_tokens(self) -> int:
        """
        Returns the maximum number of tokens for a given model

        Returns:
            int: The maximum token count.
        """
        return self._context_size

    @abstractmethod
    def run_llm_batched(
        self, prompts: List[Union[str, List[Dict[str, str]]]]
    ) -> List[Tuple[str, int]]:
        """
        Abstract method to run the target language model with a batch of prompts.

        Args:
            prompts (List[Union[str, List[Dict[str, str]]]): The list of prompts to be processed by the model.

        Returns:
            List[Tuple[str, int]]: A list of tuple objects containing the text responses and the number of tokens in the responses.
        """
        pass

    @abstractmethod
    def run_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> Tuple[str, int]:
        """
        Abstract method to run the target language model with a passed in prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt to be processed by the model.

        Returns:
            Tuple[str, int]: A tuple object containing the text response and the number of tokens in the response.
        """
        pass

    @abstractmethod
    def create_prompt_batched(
        self, results: List[Result], rank_start: int, rank_end: int, batch_size: int
    ) -> List[Tuple[Union[str, List[Dict[str, str]]], int]]:
        """
        Abstract method to create a batch of prompts based on the results and given ranking range.

        Args:
            results (List[Result]): The list of result objects containing data for prompt generation.
            rank_start (int): The starting rank for prompt generation.
            rank_end (int): The ending rank for prompt generation.

        Returns:
            Tuple[List[Union[str, List[Dict[str, str]]], List[int]]: A tuple object containing the list of generated prompts and the list of number of tokens in the generated prompts.
        """
        pass

    @abstractmethod
    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[Union[str, List[Dict[str, str]]], int]:
        """
        Abstract method to create a prompt based on the result and given ranking range.

        Args:
            result (Result): The result object containing data for prompt generation.
            rank_start (int): The starting rank for prompt generation.
            rank_end (int): The ending rank for prompt generation.

        Returns:
            Tuple[Union[str, List[Dict[str, str]]], int]: A tuple object containing the generated prompt and the number of tokens in the generated prompt.
        """
        pass

    @abstractmethod
    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """
        Abstract method to calculate the number of tokens contained in the given prompt.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt for which to compute the token count for.

        Returns:
            int: The number of tokens in the given prompt.
        """
        pass

    @abstractmethod
    def cost_per_1k_token(self, input_token: bool) -> float:
        """
        Abstract method to calculate the cost per 1,000 tokens for the target language model.

        Args:
            input_token (bool): Flag to indicate if the cost is for input tokens or output tokens.

        Returns:
            float: The cost per 1,000 tokens.
        """
        pass

    @abstractmethod
    def num_output_tokens(self) -> int:
        """
        Abstract method to estimate the number of tokens in the model's output, constrained by max tokens for the target language model.

        Returns:
            int: The estimated number of output tokens.
        """
        pass

    def permutation_pipeline_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ) -> List[Result]:
        """
        Runs the permutation pipeline on a batch of result objects within the passed in rank range.

        Args:
            results (List[Result]): The list of result objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            List[Result]: The list of processed result objects after applying permutation.
        """
        prompts = []
        prompts = self.create_prompt_batched(
            results, rank_start, rank_end, batch_size=32
        )
        output_passages_num = rank_end - rank_start

        time_start = time.time()
        batched_results = self.run_llm_batched([prompt for prompt, _ in prompts], output_passages_num=output_passages_num)
        time_cost = time.time() - time_start

        rerank_detail_batch = []
        for index, (result, (prompt, in_token_count)) in enumerate(zip(results, prompts)):
            permutation, out_token_count = batched_results[index]
            rerank_detail = {
                'qid': result.query.qid,
                'query_text': result.query.text,
                'start_idx': rank_start,
                'end_idx': rank_end - 1,
                'passage_ids': [candidate.docid for candidate in result.candidates[rank_start: rank_end]],
                'llm_output': permutation
            }
            rerank_detail_batch.append(rerank_detail)
            ranking_exec_info = RankingExecInfo(
                prompt, permutation, in_token_count, out_token_count
            )
            if result.ranking_exec_summary is None:
                result.ranking_exec_summary = []
            result.ranking_exec_summary.append(ranking_exec_info)
            result = self.receive_permutation(result, permutation, rank_start, rank_end)

        return results, time_cost, rerank_detail_batch

    def permutation_pipeline(
        self,
        result: Result,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
        populate_exec_summary: bool = True,
        num_beams: int = None
    ) -> Result:
        """
        Runs the permutation pipeline on the passed in result set within the passed in rank range.

        Args:
            result (Result): The result object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The processed result object after applying permutation.
        """
        prompt, in_token_count = self.create_prompt(result, rank_start, rank_end)
        output_passages_num = rank_end - rank_start
        time_start = time.time()
        if self._model in ['gpt-4o-mini', 'gpt-4o-2024-08-06', 'deepseek-chat', 'deepseek-reasoner']:
            permutation, in_token_count, out_token_count, reasoning, request_success = self.run_llm(prompt, output_passages_num=output_passages_num)
        else:
            permutation, in_token_count, out_token_count, request_success = self.run_llm(prompt, output_passages_num=output_passages_num, num_beams=num_beams)
        time_cost = time.time() - time_start
        rerank_detail = {
            'start_idx': rank_start,
            'end_idx': rank_end - 1,
            'passage_ids': [candidate.docid for candidate in result.candidates[rank_start: rank_end]],
            'llm_output': permutation,
            'request_success': request_success,
        }
        if self._model in ['deepseek-reasoner']:
            rerank_detail['reasoning'] = reasoning
        
        if populate_exec_summary:
            ranking_exec_info = RankingExecInfo(
                prompt, permutation, in_token_count, out_token_count
            )
            result.ranking_exec_summary.append(ranking_exec_info)
        if num_beams is None:
            result = self.receive_permutation(result, permutation, rank_start, rank_end)
        else: 
            result = [self.receive_permutation(copy.deepcopy(result), permutation_single, rank_start, rank_end) for permutation_single in permutation]
        return result, time_cost, rerank_detail

    def shuffle_and_rescore(
        self, rerank_results: List[Result], rank_start: int, rank_end: int
    ):
        """
        Shuffles candidates between rank_start and rank_end, and rescales scores based on new rank.

        Args:
            rerank_results (List[Result]): List of Result objects to process.
            rank_start (int): Start index for ranking.
            rank_end (int): End index for ranking.
        """
        for rerank_result in rerank_results:
            # Shuffle rerank_result hits between rank_start and rank_end
            rerank_result.candidates[rank_start:rank_end] = random.sample(
                rerank_result.candidates[rank_start:rank_end],
                len(rerank_result.candidates[rank_start:rank_end]),
            )
            # Rescore all candidates with 1/rank
            for i, candidate in enumerate(rerank_result.candidates):
                candidate.score = 1.0 / (i + 1)

    def sliding_windows_batched(
        self,
        requests: List[Request],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ) -> List[Result]:
        """
        Applies the sliding window algorithm to the reranking process for a batch of result objects.
        Args:
            retrieved_results (List[Request]): The list of request objects to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.
        Returns:
            List[Result]: The list of result objects after applying the sliding window technique.
        """
        rerank_results = [
            Result(
                query=copy.deepcopy(request.query),
                candidates=copy.deepcopy(request.candidates),
                ranking_exec_summary=[],
            )
            for request in requests
        ]
        if shuffle_candidates:
            self.shuffle_and_rescore(rerank_results, rank_start, rank_end)
        windows_end = rank_end
        windows_start = rank_end - window_size
        total_time_cost = 0
        rerank_details_batch = [[] for i in range(len(rerank_results))]
        # windows_end > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # windows_start + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        while windows_end > rank_start and windows_start + step != rank_start:
            logger.info(f"windows_start: {windows_start}, windows_end: {windows_end}")
            windows_start = max(windows_start, rank_start)
            rerank_results, time_cost, rerank_detail_batch = self.permutation_pipeline_batched(
                rerank_results, windows_start, windows_end, logging
            )
            for i in range(len(rerank_results)):
                rerank_details_batch[i].append(rerank_detail_batch[i])

            windows_end = windows_end - step
            windows_start = windows_start - step
            total_time_cost += time_cost
        return rerank_results, total_time_cost, rerank_details_batch

    def sliding_windows(
        self,
        request: Request,
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
        shuffle_candidates: bool = False,
        logging: bool = False,
        populate_exec_summary: bool = True,
        num_beams: int = None,
    ) -> Result:
        """
        Applies the sliding window algorithm to the reranking process.

        Args:
            request (Request): The request object to process.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.
            shuffle_candidates (bool, optional): Flag to shuffle candidates before processing. Defaults to False.
            logging (bool, optional): Flag to enable logging of operations. Defaults to False.

        Returns:
            Result: The result object after applying the sliding window technique.
        """
        rerank_result = Result(
            query=copy.deepcopy(request.query),
            candidates=copy.deepcopy(request.candidates),
            ranking_exec_summary=[],
        )
        if shuffle_candidates:
            self.shuffle_and_rescore([rerank_result], rank_start, rank_end)
        windows_end = rank_end
        windows_start = rank_end - window_size

        total_time_cost = 0
        rerank_details = []
        while windows_end > rank_start and windows_start + step != rank_start:
            windows_start = max(windows_start, rank_start)
            rerank_result, time_cost, rerank_detail = self.permutation_pipeline(
                                                                rerank_result,
                                                                windows_start,
                                                                windows_end,
                                                                logging,
                                                                populate_exec_summary=populate_exec_summary,
                                                                num_beams=num_beams
                                                            ) # note that if num_beams is not None, the rerank_result will be a List[Result] returned by beam search
            rerank_details.append(rerank_detail)
            windows_end = windows_end - step
            windows_start = windows_start - step
            total_time_cost += time_cost
        return rerank_result, total_time_cost, rerank_details

    def get_ranking_cost_upperbound(
        self, num_q: int, rank_start: int, rank_end: int, window_size: int, step: int
    ) -> Tuple[float, int]:
        """
        Calculates the upper bound of the ranking cost for a given set of parameters.

        Args:
            num_q (int): The number of queries.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the cost and the total number of tokens used (input tokens + output tokens).
        """
        # For every prompt generated for every query assume the max context size is used.
        num_promt = (rank_end - rank_start - window_size) / step + 1
        input_token_count = (
            num_q * num_promt * (self._context_size - self.num_output_tokens())
        )
        output_token_count = num_q * num_promt * self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def get_ranking_cost(
        self,
        retrieved_results: List[Request],
        rank_start: int,
        rank_end: int,
        window_size: int,
        step: int,
    ) -> Tuple[float, int]:
        """
        Calculates the ranking cost based on actual token counts from generated prompts.

        Args:
            retrieved_results (List[Request]): A list of retrieved results for processing.
            rank_start (int): The start index for ranking.
            rank_end (int): The end index for ranking.
            window_size (int): The size of each sliding window.
            step (int): The step size for moving the window.

        Returns:
            Tuple[float, int]: A tuple object containing the calculated cost and the total number of tokens used (input tokens + output tokens).
        """
        input_token_count = 0
        output_token_count = 0
        # Go through the retrieval result using the sliding window and count the number of tokens for generated prompts.
        # This is an estimated cost analysis since the actual prompts' length will depend on the ranking.
        for result in tqdm(retrieved_results):
            windows_end = rank_end
            windows_start = rank_end - window_size
            while windows_start >= rank_start:
                windows_start = max(windows_start, rank_start)
                prompt, _ = self.create_prompt(result, windows_start, windows_end)
                input_token_count += self.get_num_tokens(prompt)
                windows_end = windows_end - step
                windows_start = windows_start - step
                output_token_count += self.num_output_tokens()
        cost = (
            input_token_count * self.cost_per_1k_token(input_token=True)
            + output_token_count * self.cost_per_1k_token(input_token=False)
        ) / 1000.0
        return (cost, input_token_count + output_token_count)

    def _clean_response(self, response: str) -> str:
        if self.prompt_mode in [str(PromptMode.RANK_GPT_qwen3), str(PromptMode.RANK_GPT_reasoning)]: # split the response only when the model is not black-box model
            pattern = rf'{self.prompt_info["pattern"]}'
            match = re.search(pattern, response.lower(), re.DOTALL)
            if match:
                response = match.group(1).strip()
            else:
                if '<answer>' in response:
                    response = response.split('<answer>')[-1]
                else:
                    print('the reasoning length is too long.')
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def _remove_duplicate(self, response: List[int]) -> List[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def receive_permutation(
        self, result: Result, permutation: str, rank_start: int, rank_end: int
    ) -> Result:
        """
        Processes and applies a permutation to the ranking results.

        This function takes a permutation string, representing the new order of items,
        and applies it to a subset of the ranking results. It adjusts the ranks and scores in the
        'result' object based on this permutation.

        Args:
            result (Result): The result object containing the initial ranking results.
            permutation (str): A string representing the new order of items.
                            Each item in the string should correspond to a rank in the results.
            rank_start (int): The starting index of the range in the results to which the permutation is applied.
            rank_end (int): The ending index of the range in the results to which the permutation is applied.

        Returns:
            Result: The updated result object with the new ranking order applied.

        Note:
            This function assumes that the permutation string is a sequence of integers separated by spaces.
            Each integer in the permutation string corresponds to a 1-based index in the ranking results.
            The function first normalizes these to 0-based indices, removes duplicates, and then reorders
            the items in the specified range of the 'result.candidates' list according to the permutation.
            Items not mentioned in the permutation string remain in their original sequence but are moved after
            the permuted items.
        """
        response = self._clean_response(permutation)
        try:
            response = [int(x) - 1 for x in response.split()]
        except ValueError:
            print('unexcepted output!')
            return result

        response = self._remove_duplicate(response)
        cut_range = copy.deepcopy(result.candidates[rank_start:rank_end])
        original_rank = [tt for tt in range(len(cut_range))]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response]
        for j, x in enumerate(response): # lwh
            result.candidates[rank_start + j] = copy.deepcopy(cut_range[x])

            result.candidates[rank_start + j].score = cut_range[j].score
        return result

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def convert_doc_to_prompt_content(self, doc: Dict[str, Any], max_length: int) -> str:
        if "text" in doc:
            content = doc["text"]
        elif "segment" in doc:
            content = doc["segment"]
        elif "contents" in doc:
            content = doc["contents"]
        elif "body" in doc:
            content = doc["body"]
        else:
            content = doc["passage"]
        if "title" in doc and doc["title"]:
            content = "Title: " + doc["title"] + " " + "Content: " + content
        content = content.strip()
        content = fix_text(content)
        # For Japanese should cut by character: content = content[:int(max_length)]
        # content = " ".join(content.split()[: int(max_length)])
        content = " ".join(content.split()[: int(max_length)])
        return self._replace_number(content)
