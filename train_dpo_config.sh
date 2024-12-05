python dpo_lm.py \
    --datasets_names=hh \
    --model_path=/home/tl2020/micro/trained_models/lm/micro_model_sft_4k \
    --loss_name=dpo \
    --loss_beta=0.1 \
    --total_batch_size=128 \
    --sample_during_eval=true
