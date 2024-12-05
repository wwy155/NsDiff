export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/TimeDiff.py \
   --dataset_type="ExchangeRate" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=48 \
   --windows=48 \
   runs --seeds='[3]'
