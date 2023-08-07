PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0,1,2,3 python my_demo3.py \
    --model_name_or_path /home/linyiming/ChatGLM2-6B/ptuning/6b/chatglm2-6b \
    --ptuning_checkpoint /home/linyiming/ChatGLM2-6B/ptuning/output/chat-chatglm2-6b-pt/checkpoint-1000 \
    --pre_seq_len $PRE_SEQ_LEN

