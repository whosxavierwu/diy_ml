source /data/home/guanggao/wuzewei/venv_pytorch/bin/activate
PYTHONIOENCODING=utf-8 python run_classifier.py \
 --task_name sentiment \
 --do_eval \
 --data_dir ./data/ \
 --bert_model ./trained_models/ \
 --output_dir ./result \
 --num_train_epochs 5 \
 --learning_rate 2e-5 \
 --train_batch_size 6 \
 --max_seq_length 512 \
 --eval_batch_size 8 \

