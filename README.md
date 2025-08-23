<div align=center>
<img src="https://8421bcd.oss-cn-beijing.aliyuncs.com/img/e7634ead8cbe41e0aa55f34334327fb3.png" width="150px">
</div>
<h1 align="center"> ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability</a></h1>

<div align="center">
<a href="https://arxiv.org/pdf/2508.07050" target="_blank"><img src=https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/papers/2508.07050" target="_blank"><img src=https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface></a>
<a href="https://modelscope.cn/collections/ReasonRank-14a53a35707a46" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green.svg"></a>
<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Static Badge" src="https://img.shields.io/badge/Python-3.10+-blue.svg"></a>
</div>
<p align="center">
🤗 <a href="https://huggingface.co/liuwenhan/reasonrank-7B" target="_blank">reasonrank-7B</a> ｜
🤗 <a href="https://huggingface.co/liuwenhan/reasonrank-32B" target="_blank">reasonrank-32B</a> 
  </p>
<p align="center">
🤗 <a href="https://huggingface.co/datasets/liuwenhan/reasonrank_data_13k" target="_blank">reasonrank_data_13k</a> ｜
🤗 <a href="https://huggingface.co/datasets/liuwenhan/reasonrank_data_sft" target="_blank">reasonrank_data_sft</a> ｜
🤗 <a href="https://huggingface.co/datasets/liuwenhan/reasonrank_data_rl" target="_blank">reasonrank_data_rl</a>
</p>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub.</h5>

## 📣 Latest News
- **[Aug 16, 2025]**:🏆 Our ReasonRank (32B) has achieved **SOTA performance 42.85** on **[R2MED leaderboard](https://r2med.github.io/)**!
- **[Aug 15, 2025]**:🔥 We released our datasets and models on **[ModelScope](https://modelscope.cn/collections/ReasonRank-14a53a35707a46)**!
- **[Aug 9, 2025]**: 🏆 Our ReasonRank (32B) has achieved **SOTA performance 40.8** on **[BRIGHT leaderboard](https://brightbenchmark.github.io/)**!
- **[Aug 9, 2025]**: 📄 We uploaded our paper to the **[arXiv](https://arxiv.org/pdf/2508.07050)** and **[Hugging Face](https://huggingface.co/papers/2508.07050)**.
- **[Aug 9, 2025]**: 🔥 We released our **[🤗full reasonrank training data (13k)](https://huggingface.co/datasets/liuwenhan/reasonrank_data_13k)**, **[🤗cold-start SFT data](https://huggingface.co/datasets/liuwenhan/reasonrank_data_sft)** and **[🤗RL data](https://huggingface.co/datasets/liuwenhan/reasonrank_data_rl)**.
- **[Aug 9, 2025]**: 🔥 We released our reasoning-intensive reranker **[🤗reasonrank-7B](https://huggingface.co/liuwenhan/reasonrank-7B)** and **[🤗reasonrank-32B](https://huggingface.co/liuwenhan/reasonrank-32B)**.
- **[Aug 9, 2025]**: 🚀 We released our full codebase, including inference, SFT training, and RL training.

## 📋 Table of Contents

- [1. ReasonRank](#1-reasonrank)
  - [1.1 Overview](#11-overview)
  - [1.2 Overall Performance](#12-overall-performance)
- [2. The Introduction of ReasonRank Training Data](#2-the-introduction-of-reasonrank-training-data)
- [3. Quick Start](#3-quick-start)
  - [3.1 How to run ReasonRank](#31-how-to-run-reasonrank)
    - [3.1.1 Environment and Preparation](#311-environment-and-preparation)
    - [3.1.2 Inference on BRIGHT with ReasonIR](#312-inference-on-bright-with-reasonir)
    - [3.1.3 Inference on BRIGHT with Custom Retrieval Results](#313-inference-on-bright-with-custom-retrieval-results)
    - [3.1.4 Codes for Constructing our Input Prompt](#314-codes-for-constructing-our-input-prompt)
  - [3.2 Cold-Start SFT](#32-cold-start-sft)
    - [3.2.1 Environment Setup](#321-environment-setup)
    - [3.2.2 Supervised Fine-Tuning](#322-supervised-fine-tuning)
  - [3.3 Multiview reward ranking RL](#33-multiview-reward-ranking-rl)
    - [3.3.1 Environment Setup](#331-environment-setup)
    - [3.3.2 GRPO Training](#332-grpo-training)
  - [3.4 Performance of ReasonRank](#34-performance-of-reasonrank)
- [Citation](#citation)

## 1. ReasonRank

### 💡 1.1 Overview

**ReasonRank** is a **reasoning-intensive passage reranker** tailored for reasoning-intensive ranking tasks. To train it, we first design an automated reasoning-intensive training data synthesis framework and synthesize 1.3k high-quality training data.

<p align="center">
<img width="80%" alt="image" src="https://8421bcd.oss-cn-beijing.aliyuncs.com/img/image-20250809002302377.png" />
</p>



Based on the training data, we design a two-stage training approach including **cold-start SFT** and **multi-view ranking reward RL** to inject listwise ranking ability to our ReasonRank.

<p align="center">
<img width="80%" alt="image" src="https://8421bcd.oss-cn-beijing.aliyuncs.com/img/image-20250809002546838.png" />
</p>


### 📊 1.2 Overall Performance

When using ReasonIR as initial passage retriever, our ReasonRank demonstrates strong overall ranking performance on BRIGHT benchmark, while showing superior efficiency compared with pointwise reasoning-intensive reranker Rank1.

<p align="center">
<img width="50%" alt="image" src="https://8421bcd.oss-cn-beijing.aliyuncs.com/img/image-20250809003636871.png" />
</p>


Besides, when using a higher-quality retrieval results (RaDeR + BM25 hybrid, provided by [RaDeR](https://github.com/Debrup-61/RaDeR/blob/main/BRIGHT_score_files/RaDeR-gte-Qwen2-LLMq_CoT_lexical/aops/hybrid_BM25_Rader.json)), our ReasonRank (32B) achieves SOTA performance **40.8** on [BRIGHT leaderboard](https://brightbenchmark.github.io/).

## 📂 2. The Introduction of ReasonRank Training Data

An important contribution of our work is our reasoning-intensive training data ([reasonrank_data_13k](https://huggingface.co/datasets/liuwenhan/reasonrank_data_13k)). The dataset fields of ``training_data_all.jsonl`` are as follows:

#### **Dataset Fields & Descriptions**

1. **`dataset`** *(str)*
   - The dataset name of each piece of data (e.g., `"math-qa"`).
2. **`qid`** *(str)*
   - The query ID. The content is provided in ``id_query/`` directory.
3. **`initial_list`** *(List[str])*
   - The initial list of passage IDs before DeepSeek-R1 reranking. The content of each passage ID is provided in ``id_doc/`` directory.
4. **`final_list`** *(List[str])*
   - The re-ranked list of passage IDs after listwisely reranking with DeepSeek-R1.
   - Reflects the improved ranking based on reasoning-enhanced relevance scoring.
5. **`reasoning`** *(str)*
   - A  **step-by-step reasoning chain** outputted by DeepSeek-R1 while performing the listwise reranking.
6. **`relevant_docids`** *(List[str])*
   - The ids of relevant passages in ``initial_list`` mined by DeepSeek-R1. The remaining passage ids in ``initial_list`` are irrelevant ones. 
   - Note that **`relevant_docids`** are not necessarily ranked at the top of **`final_list`** by the DeepSeek-R1, which may stem from inconsistencies in DeepSeek-R1’s judgments. To address this, you can apply the **self-consistency data filtering** technique proposed in our paper to further select higher-quality data (i.e., evaluating the NDCG@10 of final_list using relevant_docids and only keep the ones with NDCG@10 higher than a threshold. Note that the threshold is set as 0.4 in our paper, you could try to increase it for constructing higher-quality data if you want.).

The statistics of dataset is shown in the figure below:
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/c04b9d1a-2f21-46f1-b23d-ad1f50d22fb8" />
</p>

#### **Example Entry**

```json
{
  "dataset": "math-qa",
  "qid": "math_1001",
  "initial_list": ["math_test_intermediate_algebra_808", "math_train_intermediate_algebra_1471", ...],
  "final_list": ["math_test_intermediate_algebra_808", "math_test_intermediate_algebra_1678", ...],
  "reasoning": "Okay, I need to rank the 20 passages based on their relevance...",
  "relevant_docids": ["math_test_intermediate_algebra_808", "math_train_intermediate_algebra_1471", "math_train_intermediate_algebra_993"]
}
```

#### **Application**

1. Training passage reranker: Given the reranked passage list, one can use our data to train a listwise reranker
2. Training passage retriever: Using the **`relevant_docids`** and the remaining irrelevant ids, one can train a passage retriever.

## ⚡ 3. Quick Start

### **📘** 3.1 How to run ReasonRank

#### 3.1.1 Environment and Preparation

##### Environment

In this step, we will describe the required packages for inferencing with ReasonRank. Please install the following packages.

```bash
# recommend:
# Python: Version >= 3.10
# CUDA: Version >= 12.0
pip install vllm==0.8.5.post1+cu121
pip install ftfy
pip install pyserini==0.20.0
pip install dacite
pip install pytrec-eval
pip install packaging
pip install rbo
pip install openai
pip install tenacity
pip install datasets
pip install faiss_gpu==1.7.3
pip install qwen_omni_utils
pip install blobfile
```

##### Preparation

**a.** After installing the necessary packages, remember to **update** the ``WORKSPACE_DIR`` and ``PROJECT_DIR`` (both should be absolute paths) in ``config.py``. These two parameters will be used both in our inference codes and training codes. Here is a recommended directory structure:

```bash
{WORKSPACE_DIR}
├── trained_models
│   ├── reasonrank-7B
│   └── reasonrank-32B
├── data
│   ├── bright
└── {PROJECT_DIR} (i.e., {WORKSPACE_DIR}/reasonrank)
    ├── run_rank_llm.sh
    └── run_rank_llm.py
    └── LLaMA-Factory
    └── ...
```

**b.** Download the [bright.zip](https://drive.google.com/file/d/1VKhHeiThCbziBbtMV-AOEfMjgFR7kYa5/view?usp=sharing) and [r2med.zip](https://drive.google.com/file/d/1yOrqNLr-tQYCTk0K23Y4uQTVuwoPv9hI/view?usp=drive_link) which contain the queries, qrel files and corpus of BRIGHT and R2MED datasets, unzip them and put them under ``{WORKSPACE_DIR}/data`` directory respectively.

**c.** Install jdk (we use jdk-11.0.8 in our work, other versions will also be okey)

#### 3.1.2 Inference on BRIGHT with ReasonIR

For running [reasonrank-7B](https://huggingface.co/liuwenhan/reasonrank-7B) or [reasonrank-32B](https://huggingface.co/liuwenhan/reasonrank-32B), please first download the corresponding model checkpoint and put it under ``{WORKSPACE_DIR}/trained_models`` directory. Then, run the following command under ``{PROJECT_DIR}``.

```shell
bash run_rank_llm.sh
```

The script ``run_rank_llm.sh`` includes the running command for both models. Note that the results of reproducing ReasonRank may vary slightly due to the randomness of sampling strategy and different versions of vllm.

#### 3.1.3 Inference on BRIGHT with Custom Retrieval Results

To inference with custom retrieval results, you need to put the TREC-format retrieval result files under directory ``runs/{dataset}/{file_name}``. The ``file_name`` of each dataset **must be the same**, such as custom.txt. Then specify the parameter ``retrieval_results_name`` as ``{file_name}``. The whole script is shown in run_rank_llm.sh file.

#### 3.1.4 Codes for Constructing our Input Prompt

The core codes for constructing our ReasonRank prompt is shown in ``create_prompt`` function in ``rerank/rank_listwise_os_llm.py`` file. **If you reproduce ReasonRank in your own project, please strictly use the same method for constructing the prompt to ensure ReasonRank's performance.**

```python
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
```

### ❄️ 3.2 Cold-Start SFT

#### 3.2.1 Environment Setup

In this step, we will describe how to perform a cold start SFT using the Llama Factory repository. Please first set up the environment for [Llama Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

#### 3.2.2 Supervised Fine-Tuning


1. Download our SFT dataset from [🤗reasonrank_data_sft](https://huggingface.co/datasets/liuwenhan/reasonrank_data_sft) and place it in `LLaMA-Factory/data/reasonrank_sft-data.json`. We have pre-define the dataset in `dataset_info.json`.
2. For full training (i.e., reasonrank-7B), complete the path information in `LLaMA-Factory/examples/train_full/qwen_full_sft.yaml`. The file content should be as follows:

```yaml
### model
model_name_or_path: {YOUR_BACKBONE_MODEL_PATH}/Qwen2.5-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: reasonrank_sft-data
template: qwen
cutoff_len: 23552
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {YOUR_MODEL_SAVE_PATH}
logging_steps: 10
# save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5e-6
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
save_strategy: epoch
```

​		After completing the information, you can fine-tune the model using the following command:

```shell
cd LLaMA-Factory
bash run_train.sh
```

3. For lora training (i.e., reasonrank-32B), complete the path information in `LLaMA-Factory/examples/train_lora/qwen_lora_sft.yaml`. The file content should be as follows:

```yaml
### model
model_name_or_path: {YOUR_BACKBONE_MODEL_PATH}/Qwen2.5-32B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 32
lora_target: all
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: reasonrank_sft-data
template: qwen
cutoff_len: 23552
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {YOUR_MODEL_SAVE_PATH}
logging_steps: 10
# save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 7.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
save_strategy: epoch
```

​		After completing the information, you can fine-tune the model using the following command:

```shell
cd LLaMA-Factory
bash run_train_lora.sh
```

---

### 🔥 3.3 Multi-reward ranking RL

In this step, we will load the cold-start model for GRPO training. We use [VERL](https://github.com/volcengine/verl) frameworks for RL training.


#### 3.3.1. Environment Setup

 you can install our additional environment as follow: 

```bash
cd verl  # we use verl==0.4.0
bash scripts/install_vllm_sglang_mcore.sh
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

#### 3.3.2. GRPO Training

> Our multi-view ranking reward is implemented in the ``verl/verl/utils/reward_score/ranking.py``. 

**a.** Remember to update the pattern file path in ``verl/verl/utils/reward_score/ranking.py`` with the absolute path of our project reasonrank.

```python
pattern = toml.load('{YOUR_PROJECT_DIR}/listwise_prompt_r1.toml')['pattern']
```

**b.** Update the ``YOUR_PROJECT_DIR`` in ``verl/scripts/merge.sh`` with the absolute path of our project reasonrank.

**c.** Download our RL dataset from [🤗reasonrank_data_rl](https://huggingface.co/datasets/liuwenhan/reasonrank_data_rl) and place the training set file and validation file in `verl/data/`. 

**d.** Run the following command to train ReasonRank (7B):

```shell
bash train_grpo.sh
```

Remember to change the ``actor_rollout_ref.model.path`` to the path of your SFT model and ``trainer.default_local_dir`` to the model saving path.

**e.** Run the following command to train ReasonRank (32B) with lora:

```shell
bash train_grpo_lora.sh
```

Remember to change the ``actor_rollout_ref.model.path`` to the path of your SFT model and ``trainer.default_local_dir`` to the model saving path.

### 📊 3.4 Performance of ReasonRank

<p align="center">
<img width="90%" alt="image" src="https://8421bcd.oss-cn-beijing.aliyuncs.com/img/image-20250810163757771.png" />
</p>

## 📄 Citation

If you find this work helpful, please cite our papers:

```bibtex
@article{liu2025reasonrank,
  title={ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability},
  author={Liu, Wenhan and Ma, Xinyu and Sun, Weiwei and Zhu, Yutao and Li, Yuchen and Yin, Dawei and Dou, Zhicheng},
  journal={arXiv preprint arXiv:2508.07050},
  year={2025}
}
```


## 🤝 Acknowledge

The inference codes and training implementation build upon [RankLLM](https://github.com/castorini/rank_llm), [Llama Factory](https://github.com/hiyouga/LLaMA-Factory) and [verl](https://github.com/volcengine/verl). Our work is based on the [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model series, and we sincerely thank the Qwen team for their outstanding contributions to the open-source community.


## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [lwh@ruc.edu.cn](lwh@ruc.edu.cn).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=8421bcd/reasonrank&type=Date)](https://www.star-history.com/#8421bcd/reasonrank&Date)
