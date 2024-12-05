export PYTHONPATH=./
export PYTHONPATH=/notebooks/pytorchtimseries:./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/DiffusionTS_nonoverlap.py \
   --dataset_type="SolarEnergy" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=24 \
   --windows=168 \
   runs --seeds='[33]'
