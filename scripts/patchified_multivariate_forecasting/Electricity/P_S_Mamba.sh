export CUDA_VISIBLE_DEVICES=0

model_name=P_S_Mamba

python -u run.py \
  --is_training 1 \
  -root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --learning_rate 0.0001 \
  --train_epochs 10\
  --d_state 2 \
  --d_ff 512\
  --itr 1

