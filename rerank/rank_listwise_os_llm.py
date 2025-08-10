import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
import traceback
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import toml
import torch
from ftfy import fix_text
from tqdm import tqdm
from transformers.generation import GenerationConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from vllm.lora.request import LoRARequest

try:
    from vllm import LLM, SamplingParams
except:
    LLM = None
    SamplingParams = None

from data import Result
from rerank.rankllm import PromptMode, RankLLM
from utils import add_prefix_prompt, add_post_prompt, convert_doc_to_prompt_content

class RankListwiseOSLLM(RankLLM):
    def __init__(
        self,
        args,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda:1",
        num_gpus: int = 1,
        window_size: int = 20,
        prompt_info_path: str = None,
        vllm_batched: bool = False,
        max_passage_length: int = 100,
    ) -> None:
        """
         Creates instance of the RankListwiseOSLLM class, an extension of RankLLM designed for performing listwise ranking of passages using
         a specified language model. Advanced configurations are supported such as GPU acceleration, variable passage
         handling, and custom system messages for generating prompts.

         Parameters:
         - model (str): Identifier for the language model to be used for ranking tasks.
         - context_size (int, optional): Maximum number of tokens that can be handled in a single prompt. Defaults to 4096.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
         - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
         the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
         by default.
         - device (str, optional): Specifies the device for model computation ('cuda' for GPU or 'cpu'). Defaults to 'cuda'.
         - num_gpus (int, optional): Number of GPUs to use for model loading and inference. Defaults to 1.
         - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
         - vllm_batched (bool, optional): Indicates whether batched inference using VLLM is leveraged. Defaults to False.

         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        """
        super().__init__(model, context_size, prompt_mode, prompt_info_path, num_few_shot_examples)
        self.args = args
        self._device = device
        self.max_passage_length = max_passage_length
        self.prompt_mode = prompt_mode
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode not in [str(PromptMode.RANK_GPT), str(PromptMode.RANK_GPT_reasoning), str(PromptMode.RANK_GPT_qwen3), str(PromptMode.RANK_GPT_rankk)]:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of {PromptMode.RANK_GPT} prompt."
            )
        # ToDo: Make repetition_penalty configurable
        if vllm_batched and LLM is None:
            raise ImportError("Please install rank-llm with `pip install rank-llm[vllm]` to use batch inference.")
        elif vllm_batched:
            if 'GPTQ' in model:
                self._llm = LLM(model, quantization="gptq", download_dir=os.getenv("HF_HOME"), gpu_memory_utilization=0.85, enforce_eager=False, tensor_parallel_size=num_gpus)
            else:
                self._llm = LLM(model, 
                                enable_lora=True if args.lora_path is not None else False,
                                max_lora_rank=args.max_lora_rank,
                                download_dir=os.getenv("HF_HOME"), 
                                gpu_memory_utilization=0.9, 
                                enforce_eager=False, 
                                tensor_parallel_size=num_gpus
                                )
            self._tokenizer = self._llm.get_tokenizer()
        else:
            if 'GPTQ' in model:
                self._llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
            else:
                self._llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(model)
            
        self._vllm_batched = vllm_batched
        self._window_size = window_size
        self.prompt_info = toml.load(prompt_info_path)
        self._output_token_estimate = None
        self.lora_request = LoRARequest("R1adapter", 1, args.lora_path) if args.lora_path is not None else None
        if args.lora_path is not None:
            print(f'loading lora from {args.lora_path}')

    def run_llm_batched(self, prompts: List[str], output_passages_num: Optional[int] = None,) -> List[Tuple[str, int]]:
        if SamplingParams is None:
            raise ImportError("Please install rank-llm with `pip install rank-llm[vllm]` to use batch inference.")
        if self.prompt_mode in [str(PromptMode.RANK_GPT_reasoning), str(PromptMode.RANK_GPT_rankk)]:
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=self.args.reasoning_maxlen + 100,
            )
        elif self.prompt_mode in [str(PromptMode.RANK_GPT_qwen3)]:
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                max_tokens=self.args.reasoning_maxlen + 100,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.num_output_tokens(output_passages_num),
                min_tokens=self.num_output_tokens(output_passages_num),
            )
        outputs = self._llm.generate(prompts, 
                                    sampling_params, 
                                    use_tqdm=True,
                                    lora_request=self.lora_request,
                                    )

        return [(output.outputs[0].text, len(output.outputs[0].token_ids)) for output in outputs]

    def run_llm(self, prompt: str, output_passages_num: Optional[int] = None, num_beams = None) -> Tuple[str, int]:
        inputs = self._tokenizer([prompt])
        inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        if self.prompt_mode not in [str(PromptMode.RANK_GPT_qwen3), str(PromptMode.RANK_GPT_reasoning), str(PromptMode.RANK_GPT_rankk)]:
            gen_cfg.max_new_tokens = self.num_output_tokens(output_passages_num)
            gen_cfg.min_new_tokens = self.num_output_tokens(output_passages_num)

        # sampling search
        gen_cfg.num_beams = 1
        gen_cfg.do_sample = True
        gen_cfg.num_return_sequences = num_beams
        gen_cfg.temperature = 1.0  # Higher values lead to more random outputs
        gen_cfg.top_k = 50         # Consider only the top_k tokens by probability
        gen_cfg.top_p = 0.95       # Use nucleus sampling (top-p)

        if num_beams is None:
            output_ids = self._llm.generate(**inputs, generation_config=gen_cfg)
            if self._llm.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
            outputs = self._tokenizer.decode(output_ids, skip_special_tokens=False, spaces_between_special_tokens=False)
            return outputs, output_ids.size(0)
        else:
            output_ids_beam = self._llm.generate(**inputs, generation_config=gen_cfg)
            output_ids_beam = [output_ids if self._llm.config.is_encoder_decoder else output_ids[len(inputs["input_ids"][0]):] for output_ids in output_ids_beam]
            outputs = self._tokenizer.batch_decode(output_ids_beam, skip_special_tokens=False, spaces_between_special_tokens=False)
            return outputs, output_ids_beam[0].size(0)

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            if self.prompt_mode in [str(PromptMode.RANK_GPT), str(PromptMode.RANK_GPT_qwen3), str(PromptMode.RANK_GPT_reasoning), str(PromptMode.RANK_GPT_rankk)] :
                _output_token_estimate = len(self._tokenizer.encode(" > ".join([f"[{i+1}]" for i in range(current_window_size)])))
            else:
                raise ValueError('not supported prompt_mode!')
            # print('_output_token_estimate', _output_token_estimate)
            if self._output_token_estimate is None and self._window_size == current_window_size:
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate


    def create_prompt(self, result: Result, rank_start: int, rank_end: int) -> Tuple[str, int]:
        query = result.query.text
        qid = result.query.qid
        query = self._replace_number(query).strip()
        num = len(result.candidates[rank_start:rank_end])
        max_length = self.max_passage_length

        #################### core codes for constructing the input ####################
        messages = []
        if self.args.prompt_mode == str(PromptMode.RANK_GPT_reasoning):  # for non-reasoning model such as qwen2.5
            messages.append({"role": "system", "content": self.prompt_info['system_prompt_reasoning']})
        elif self.args.prompt_mode in [str(PromptMode.RANK_GPT), str(PromptMode.RANK_GPT_qwen3)]:
            messages.append({"role": "system", "content": self.prompt_info['system_prompt']})

        prefix = add_prefix_prompt(promptmode=self.prompt_mode, query=query, num=num)
        rank = 0
        input_context = f"{prefix}\n"
        for cand in result.candidates[rank_start:rank_end]:
            rank += 1
            content = convert_doc_to_prompt_content(self._tokenizer, cand.doc, max_length, truncate_by_word=False)
            input_context += f"[{rank}] {content}\n"

        input_context += add_post_prompt(promptmode=self.prompt_mode, query=query, num=num)
        messages.append({"role": "user", "content": input_context})
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = fix_text(prompt)
        #################### core codes for constructing the input ####################

        num_tokens = self.get_num_tokens(prompt)
        return prompt, num_tokens

    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
        batch_size: int = 32,
    ) -> List[Tuple[str, int]]:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, rank_start, rank_end),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts


    def _add_few_shot_examples(self, conv):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
        return conv

    def get_num_tokens(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
