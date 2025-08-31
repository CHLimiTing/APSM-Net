root_path_name=./data/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

seed=2025

seq_len=512

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
    --alpha 1.0 \
    --top_k 2 \
    --target_patch_len 48 \
    --d_core $seq_len \
    --norm False \
    --layernorm False \
    --dropout 0.1 \
    --train_epochs 20 \
    --batch_size 128 \
    --learning_rate 0.00005 \
    --early_stopping 3 \
    --result_path result.csv
done
