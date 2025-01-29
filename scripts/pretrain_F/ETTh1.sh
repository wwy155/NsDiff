export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/pretrain_f.py \
   --dataset_type="ETTh1" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   --epochs=10 \
   --patience=10 \
   runs --seeds='[10, 2, 3]'
