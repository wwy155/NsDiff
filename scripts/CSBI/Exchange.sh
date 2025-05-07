export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/CSBI.py \
   --dataset_type="ExchangeRate" \
   --device="cuda:0" \
   --batch_size=2 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[3]'
