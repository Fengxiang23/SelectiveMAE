torchrun --nproc_per_node=8 --nnodes 1 --master_port 14884 \
--dataset 'aid' --model 'vit_base_patch16' --postfix 'sota' \
--batch_size 1024 --epochs 100 --warmup_epochs 5 \
--blr 1e-3  --weight_decay 0.05 --split 19  --tag 0 --exp_num 1 \
--data_path   ''     \
--finetune 'checkpoint.pth'

