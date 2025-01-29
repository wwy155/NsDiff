export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/pretrain_f.py \
   --dataset_type="ETTm2" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=24 \
   --windows=168 \
   --epochs=50 \
   --patience=10 \
   runs --seeds='[1, 2, 3]'
