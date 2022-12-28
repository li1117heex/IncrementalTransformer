# Cache-Assist QA: Faster Online QA by Caching Passage Representations to Assist Question Understanding

This repo contains codes of Cache-Assist QA.

## Setup

Run `python prepare_datasets.py` to download and splits datasets including SQuADv1.1, SQuADv2 , RACE and BoolQ.

## How to use

This is a example for finetuning Cache-Assist QA on SQuADV1.1 dataset with $k=m=20$. `teacher_model_path` designates model in `ro_squad_large` as teacher model for auxiliary loss and `--alpha` means the ratio of auxiliary loss.  Teacher model is the finetuned RoBERTa model of the same datasets, can be obtained by scripts like `run_qa.py`. For SQuADv2 dataset, `--version_2_with_negative `should be set for unanswerable questions. For RACE and BoolQ datasets, use `run_race_cacheassistqa.py` and `run_boolq_cacheassistqa.py` respectively. 

```shell
python run_qa_cacheassistqa.py \
	--model_name_or_path roberta-large \
	--dataset_name squadsplit \
	--do_train \
	--do_eval \
	--do_predict \
	--per_device_train_batch_size 8 \
	--learning_rate 2e-5 \
	--num_train_epochs 2 \
	--max_seq_length 384 \
	--doc_stride 128 \
	--output_dir cacheassistqa_squad \
	--eval_steps 500 \
	--save_steps 1000 \
	--evaluation_strategy steps \
	--save_strategy steps \
	--overwrite_output_dir \
	--teacher_model_path ro_squad_large \
	--alpha 0.9 \
	--weight_decay 0.01 \
	--warmup_ratio 0.06 \
	--decomposed_layers 0 \
	--assist_layers 20 \
	--full_layers 4
```

## Requirements

```
pytorch==1.10.2
transformers==4.20.1
datasets==1.11.0
tokenizers=0.11.4
```



