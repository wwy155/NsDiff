export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff.py \
   --dataset_type="ETTm1" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=24 \
   --windows=168 \
   --rolling_length=24 \
   --epochs=50 \
   --patience=10 \
   runs --seeds='[1, 2, 3]'
