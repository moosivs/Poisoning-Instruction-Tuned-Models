wandb login 9773efedf555cf56a523a48d1f32e7b323f04c91

export PYTHONPATH=${PWD}/src/

python scripts/llama_chat_finetune.py
# torchrun --nproc_per_node=2 scripts/llama_chat_finetune.py