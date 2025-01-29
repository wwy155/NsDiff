export PYTHONPATH=./:/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/D3VAE.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="Weather" \
   --device="cuda:6" \
   --batch_size=8 \
   --horizon=1 \
   --pred_len=24 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'

# python3 ./src/experiments/CSDI.py \
#    config_wandb --project=3108Diffusion \
#    --dataset_type="Traffic" \
#    --device="cuda:0" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len=24 \
#    --windows=168 \
#    --epochs=100   \
#    runs --seeds='[3]'
