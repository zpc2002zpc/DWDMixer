# export CUDA_VISIBLE_DEVICES=0

model_name=DWDMixer

seq_len=96
down_sampling_window=2
learning_rate=0.01
d_ff=32
batch_size=16
wt_level=2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --channel_independence 1 \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --batch_size $batch_size  \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window\
  --wt_level $wt_level


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 8 \
  --channel_independence 1 \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --batch_size $batch_size  \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window\
  --wt_level $wt_level


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 16 \
  --channel_independence 1 \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --batch_size $batch_size  \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window\
  --wt_level $wt_level


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 8 \
  --channel_independence 1 \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --batch_size $batch_size  \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window\
  --wt_level $wt_level
