本仓库包含了增量式Transformer编码器模型的代码与其训练脚本，可以在QA任务的SQuADv1.1,SQuADv2,RACE,BoolQ等数据集上训练基于RoBERTa的增量式Transformer编码器模型。`incremental_attn_p2q.py`为增量式模型的注意力模块。

## 如何使用

使用`run_qa_deformerassist_sepqp_local.py`可以加载RoBERTa预训练模型用以初始化增量式Transformer编码器模型，并在SQuAD数据集上进行fine-tune，而且其训练目标加入了蒸馏loss，可以用``--teacher_model_path`指定一个fine-tune完成的非增量式RoBERTa模型，将其输出分类软标签也作为学习目标，`--alpha`可以设置蒸馏loss的比例。`--encoder2_layers 0 --assist_layer 0 --encoder3_layers 24`为模型相关参数，详情见专利。如果数据集中部分问题没有答案，比如SQuADv2，需要设置`--version_2_with_negative`。例子如下：

```shell
python run_qa_deformerassist_sepqp_local.py --model_name_or_path roberta-large --dataset_name squadsplit --do_train --do_eval --do_predict --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir deformerassist_squad_b162e-5wudu0l10l24lseppqnew --eval_steps 500 --save_steps 1000 --evaluation_strategy steps --save_strategy steps --overwrite_output_dir --teacher_model_path ro_squad_largeb64wdwu --alpha 0 --weight_decay 0.01 --warmup_ratio 0.06 --encoder2_layers 0 --assist_layer 0 --encoder3_layers 24
```

`run_race_deformerassist_seppq_local_len.py`,`run_boolq_deformerassist_seppq_local_len.py`同理，用于RACE,BoolQ数据集:

```shell
python run_race_deformerassist_seppq_local_len.py --model_name_or_path roberta-large --do_train --do_eval --do_predict --learning_rate 1e-5 --num_train_epochs 4 --per_device_train_batch_size 8 --overwrite_output --warmup_ratio 0.06 --weight_decay 0.1 --save_steps 1000 --eval_steps 1000 --evaluation_strategy steps --save_strategy steps --output_dir deformerassist_race_teachslalpha0.9b161e-5wudu0l12l12lseppqnew --teacher_model_path ro_squad_largeb64wdwupre --alpha 0.9 --encoder2_layers 0 --assist_layers 12 --encoder3_layers 12
python run_boolq_deformerassist_seppq_local_len.py --model_name_or_path roberta-large --dataset_path boolqsplit --do_train --do_eval --do_predict --learning_rate 1e-5 --num_train_epochs 2 --per_device_train_batch_size 8 --overwrite_output --warmup_ratio 0.06 --weight_decay 0.1 --save_steps 200 --eval_steps 200 --evaluation_strategy steps --save_strategy steps --output_dir deformerassist_boolq_teachhsalpha0.9b161e-5wudu0l23l1lseppq --teacher_model_path ro_boolq_b162e-5wudu --alpha 0.9 --encoder2_layers 0 --assist_layers 23 --encoder3_layers 1
```

## requirements

```
pytorch
transformers
datasets
```



