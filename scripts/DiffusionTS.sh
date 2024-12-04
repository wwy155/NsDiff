export PYTHONPATH=./
export PYTHONPATH=/notebooks/pytorchtimseries:./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/DiffusionTS.py \
   --dataset_type="ExchangeRate" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=24 \
   runs --seeds='[3]'
