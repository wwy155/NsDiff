export PYTHONPATH=./:/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/D3VAE.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="Electricity" \
   --device="cuda:6" \
   --num_preprocess_cells=1 \
   --batch_size=4 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'