import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
import openai
import tiktoken
from tqdm import tqdm
from data import Result
from rerank.rankllm import PromptMode, RankLLM
from openai import OpenAI
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from transformers import AutoTokenizer, AutoModel
from utils import convert_doc_to_prompt_content, replace_number
import os
from config import WORKSPACE_DIR


class SafeOpenai(RankLLM):
    def __init__(
        self,
        args,
        model: str,
        context_size: int,
        resource: str = '',
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        window_size: int = 20,
        prompt_info_path: str = None,
        keys=None,
        key_start_id=None,
        proxy=None,
        api_type: str = None,
        api_base: str = None,
        api_version: str = None,
        max_passage_length: int = 100,
    ) -> None:
        """
        Creates instance of the SafeOpenai class, a specialized version of RankLLM designed for safely handling OpenAI API calls with
        support for key cycling, proxy configuration, and Azure AI conditional integration.

        Parameters:
        - model (str): The model identifier for the LLM (model identifier information can be found via OpenAI's model lists).
        - context_size (int): The maximum number of tokens that the model can handle in a single request.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
        - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
        the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
        by default.
        - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
        - keys (Union[List[str], str], optional): A list of OpenAI API keys or a single OpenAI API key.
        - key_start_id (int, optional): The starting index for the OpenAI API key cycle.
        - proxy (str, optional): The proxy configuration for OpenAI API calls.
        - api_type (str, optional): The type of API service, if using Azure AI as the backend.
        - api_base (str, optional): The base URL for the API, applicable when using Azure AI.
        - api_version (str, optional): The API version, necessary for Azure AI integration.

        Raises:
        - ValueError: If an unsupported prompt mode is provided or if no OpenAI API keys / invalid OpenAI API keys are supplied.

        Note:
        - This class supports cycling between multiple OpenAI API keys to distribute quota usage or handle rate limiting.
        - Azure AI integration is depends on the presence of `api_type`, `api_base`, and `api_version`.
        """
        super().__init__(model, context_size, prompt_mode, prompt_info_path, num_few_shot_examples)
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("Please provide OpenAI Keys.")
        if prompt_mode not in [
            str(PromptMode.RANK_GPT),
            str(PromptMode.RANK_GPT_reasoning),
            str(PromptMode.RANK_GPT_APEER),
            str(PromptMode.LRL),
        ]:
            raise ValueError(f"unsupported prompt mode for GPT models: {prompt_mode}, expected {PromptMode.RANK_GPT}, {PromptMode.RANK_GPT_reasoning}, {PromptMode.RANK_GPT_APEER} or {PromptMode.LRL}.")
        self.args = args
        self.resource = resource
        self.prompt_mode = prompt_mode
        self._window_size = window_size
        self._output_token_estimate = None
        self._keys = keys
        self._cur_key_id = key_start_id or 0
        self._cur_key_id = self._cur_key_id % len(self._keys)
        self.tokenizer = AutoTokenizer.from_pretrained(f'Qwen2.5-7B-Instruct')
        self.max_passage_length = max_passage_length

        openai.proxy = proxy
        openai.api_key = self._keys[self._cur_key_id]
        self.use_azure_ai = False

        if all([api_type, api_base, api_version]):
            # See https://learn.microsoft.com/en-US/azure/ai-services/openai/reference for list of supported versions
            openai.api_version = api_version
            openai.api_type = api_type
            openai.api_base = api_base
            self.use_azure_ai = True

    class CompletionMode(Enum):
        UNSPECIFIED = 0
        CHAT = 1
        TEXT = 2

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def req(self, model_name, messages, max_tokens):
        client = OpenAI(api_key=openai.api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens
        )
        # response = completion.choices[0].message.content
        return completion


    def run_llm(self, prompt: Union[str, List[Dict[str, str]]], output_passages_num: Optional[int] = None,) -> Tuple[str, int]:
        request_success = True
        try:
            completion = self.req(
                model_name=self._model, 
                messages=prompt, 
                # max_tokens=self.num_output_tokens(output_passages_num)
                max_tokens=500
                )
        except:
            print(completion)
            request_success = False
        if request_success:
            content = completion.choices[0].message.content
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            # reasoning = completion.choices[0].message.reasoning_content
            return content, prompt_tokens, completion_tokens, request_success
        else:
            return '', 0, 0, request_success


    def run_llm_batched(self, prompts: List[List[Dict]], output_passages_num: Optional[int] = None,) -> List[Tuple[str, int]]:
        outputs = []
        for prompt in tqdm(prompts):
            try:
                completion = self.req(
                    model_name=self._model, 
                    messages=prompt, 
                    # max_tokens=self.num_output_tokens(output_passages_num)
                    max_tokens=500
                    )
            except:
                print(completion)
                exit()
            content = completion.choices[0].message.content
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            outputs.append((content, completion_tokens))
        return outputs

    def _get_prefix_for_rank_gpt_prompt(self, query: str, num: int) -> List[Dict[str, str]]:
        if 'deepseek' in self._model:  # for R1
            return [
                {
                    "role": "user",
                    "content": (
                        f"You are an intelligent assistant that can rank passages based on their relevance to the query. "
                        f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."
                    )
                },
                {"role": "assistant", "content": "Okay, please provide the passages."},
            ]
        else:
            return [
                {
                    "role": "system",
                    "content": "You are an intelligent assistant that can rank passages based on their relevance to the query.",
                },
                {
                    "role": "user",
                    "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
                },
                {"role": "assistant", "content": "Okay, please provide the passages."},
            ]


    def _get_suffix_for_rank_gpt_prompt(self, query: str, num: int) -> str:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [2] > [1]. Only response the ranking results, do not say any word or explain."

    def _get_suffix_for_rank_gpt_reasoning_prompt(self, query: str, num: int) -> str:
        return f"Search Query: {query}. \nFirst think and then rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The answer format should be [] > [], e.g., [2] > [1]."

    def _get_prefix_for_rank_gpt_apeer_prompt(self, query: str, num: int) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "As RankGPT, your task is to evaluate and rank unique passages based on their relevance and accuracy to a given query. Prioritize passages that directly address the query and provide detailed, correct answers. Ignore factors such as length, complexity, or writing style unless they seriously hinder readability.",
            },
            {
                "role": "user",
                "content": f"In response to the query: [querystart] {query} [queryend], rank the passages. Ignore aspects like length, complexity, or writing style, and concentrate on passages that provide a comprehensive understanding of the query. Take into account any inaccuracies or vagueness in the passages when determining their relevance.",
            },
        ]

    def _get_suffix_for_rank_gpt_apeer_prompt(self, query: str, num: int) -> str:
        return f"Given the query: [querystart] {query} [queryend], produce a succinct and clear ranking of all passages, from most to least relevant, using their identifiers. The format should be [rankstart] [most relevant passage ID] > [next most relevant passage ID] > ... > [least relevant passage ID] [rankend]. Refrain from including any additional commentary or explanations in your ranking."


    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            try:
                encoder = tiktoken.get_encoding(self._model)
            except:
                encoder = tiktoken.get_encoding("cl100k_base")

            _output_token_estimate = (
                len(
                    encoder.encode(
                        " > ".join([f"[{i+1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate

    def create_prompt(self, result: Result, rank_start: int, rank_end: int) -> Tuple[List[Dict[str, str]], int]:
        if self.prompt_mode in [str(PromptMode.RANK_GPT), str(PromptMode.RANK_GPT_reasoning), str(PromptMode.RANK_GPT_APEER)]:
            return self.create_rank_gpt_prompt(result, rank_start, rank_end)
        else:
            return self.create_LRL_prompt(result, rank_start, rank_end)

    def create_prompt_batched(self, results: List[Result], rank_start: int, rank_end: int, batch_size: int = 32,) -> List[Tuple[str, int]]:
        prompts = []
        for result in results:
            prompt = self.create_rank_gpt_prompt(result, rank_start, rank_end)
            prompts.append(prompt)
        return prompts
        
    def create_rank_gpt_prompt(self, result: Result, rank_start: int, rank_end: int) -> Tuple[List[Dict[str, str]], int]:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])
        max_length = self.max_passage_length

        if self.prompt_mode == str(PromptMode.RANK_GPT):
            messages = self._get_prefix_for_rank_gpt_prompt(query, num)
        # elif self.prompt_mode == str(PromptMode.RANK_GPT_reasoning):
        #     messages = self._get_prefix_for_rank_gpt_reasoning_prompt(query, num)
        else:
            messages = self._get_prefix_for_rank_gpt_apeer_prompt(query, num)
        rank = 0
        for cand in result.candidates[rank_start:rank_end]:
            rank += 1
            content = convert_doc_to_prompt_content(self.tokenizer, cand.doc, max_length)
            # content = self.convert_doc_to_prompt_content(cand.doc, max_length)
            if self.prompt_mode in [str(PromptMode.RANK_GPT), str(PromptMode.RANK_GPT_reasoning)]:
                messages.append({"role": "user", "content": f"[{rank}] {content}",})
                messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
            else:
                messages[-1]["content"] += f"\n[{rank}] {content}"

        if self.prompt_mode == str(PromptMode.RANK_GPT):
            messages.append({"role": "user", "content": self._get_suffix_for_rank_gpt_prompt(query, num)})
        elif self.prompt_mode == str(PromptMode.RANK_GPT_reasoning):
            messages.append({"role": "user", "content": self._get_suffix_for_rank_gpt_reasoning_prompt(query, num)})
        else:
            messages[-1]["content"] += f"\n{self._get_suffix_for_rank_gpt_apeer_prompt(query, num)}"

        return messages, -1


    def create_LRL_prompt(self, result: Result, rank_start: int, rank_end: int) -> Tuple[List[Dict[str, str]], int]:
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])
        max_length = 300 * (20 / (rank_end - rank_start))
        psg_ids = []
        while True:
            message = "Sort the list PASSAGES by how good each text answers the QUESTION (in descending order of relevance).\n"
            rank = 0
            for cand in result.candidates[rank_start:rank_end]:
                rank += 1
                psg_id = f"PASSAGE{rank}"
                content = convert_doc_to_prompt_content(self.tokenizer, cand.doc, max_length)
                message += f'{psg_id} = "{self._replace_number(content)}"\n'
                psg_ids.append(psg_id)
            message += f'QUESTION = "{query}"\n'
            message += "PASSAGES = [" + ", ".join(psg_ids) + "]\n"
            message += "SORTED_PASSAGES = [\n"
            messages = [{"role": "user", "content": message}]
            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.num_output_tokens():
                break
            else:
                max_length -= max(
                    1,
                    (num_tokens - self.max_tokens() + self.num_output_tokens())
                    // ((rank_end - rank_start) * 4),
                )
        return messages, self.get_num_tokens(messages)

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """Returns the number of tokens used by a list of messages in prompt."""
        if self._model in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif self._model in ["gpt-4-0314", "gpt-4", "gpt-4o-2024-08-07"]:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message, tokens_per_name = 0, 0

        try:
            encoding = tiktoken.get_encoding('gpt-3.5-turbo')
        except:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        if isinstance(prompt, list):
            for message in prompt:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        else:
            num_tokens += len(encoding.encode(prompt))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def cost_per_1k_token(self, input_token: bool) -> float:
        # Brought in from https://openai.com/pricing on 2023-07-30
        cost_dict = {
            ("gpt-3.5", 4096): 0.0015 if input_token else 0.002,
            ("gpt-3.5", 16384): 0.003 if input_token else 0.004,
            ("gpt-4", 8192): 0.03 if input_token else 0.06,
            ("gpt-4", 32768): 0.06 if input_token else 0.12,
        }
        model_key = "gpt-3.5" if "gpt-3" in self._model else "gpt-4"
        return cost_dict[(model_key, self._context_size)]

