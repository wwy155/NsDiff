export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/D3VAE.py \
   --dataset_type="ExchangeRate" \
   --device="cuda:0" \
   --batch_size=20 \
   --horizon=1 \
   --pred_len=24 \
   --windows=48 \
   --epochs=100   \
   runs --seeds='[2]'
