root_path_name=./data/
data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=exchange_rate

seed=2025

seq_len=96

for pred_len in 96 192 336 720
do
  python -u main_lightgts_v2.py \
    --seed $seed \
    --data $root_path_name$data_path_name \
    --feature_type M \
    --target OT \
    --checkpoint_dir ./checkpoints \
    --name $model_id_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --alpha 0.0 \
    --top_k 2 \
    --target_patch_len 24 \
    --d_core $seq_len \
    --norm True \
    --layernorm True \
    --dropout 0.1 \
    --train_epochs 10 \
    --batch_size 512 \
    --learning_rate 0.0003 \
    --early_stopping 3 \
    --result_path result.csv
done
