#
Model_path=openi_model_biobert_777_dec_0.04

gpus=5,6,7
# gpus=-1
log_file=$Model_path.log

lr_bert=0.0005
# lr_bert=0.0006


lr_dec=0.04

echo $lr_bert
echo $lr_dec

python train.py  \
-task abs \
-mode train \
-bert_data_path ../biobert_entity_openi_data/radiology/radiology \
-dec_dropout 0.2  \
-model_path $Model_path \
-sep_optim true \
-lr_bert ${lr_bert} \
-lr_dec ${lr_dec} \
-save_checkpoint_steps 500 \
-batch_size 128 \
-train_steps 20000 \
-report_every 50 \
-accum_count 5 \
-use_bert_emb true \
-use_interval true \
-warmup_steps_bert 2000 \
-warmup_steps_dec 1500 \
-max_pos 512 \
-visible_gpus $gpus \
-log_file ../logs/$log_file \
-encoder bert \
-seed 777






#
Model_path2=openi_model_biobert_777_0.00045

gpus=5,6,7
# gpus=-1
log_file2=$Model_path2.log

lr_bert=0.00045
# lr_bert=0.0006


lr_dec=0.05

echo $lr_bert
echo $lr_dec

python train.py  \
-task abs \
-mode train \
-bert_data_path ../biobert_entity_openi_data/radiology/radiology \
-dec_dropout 0.2  \
-model_path $Model_path2 \
-sep_optim true \
-lr_bert ${lr_bert} \
-lr_dec ${lr_dec} \
-save_checkpoint_steps 500 \
-batch_size 128 \
-train_steps 20000 \
-report_every 50 \
-accum_count 5 \
-use_bert_emb true \
-use_interval true \
-warmup_steps_bert 2000 \
-warmup_steps_dec 1500 \
-max_pos 512 \
-visible_gpus $gpus \
-log_file ../logs/$log_file2 \
-encoder bert \
-seed 777





Model_path3=openi_model_biobert_222

gpus=5,6,7
# gpus=-1
log_file3=$Model_path3.log

lr_bert=0.0005
# lr_bert=0.0006


lr_dec=0.05

echo $lr_bert
echo $lr_dec

python train.py  \
-task abs \
-mode train \
-bert_data_path ../biobert_entity_openi_data/radiology/radiology \
-dec_dropout 0.2  \
-model_path $Model_path3 \
-sep_optim true \
-lr_bert ${lr_bert} \
-lr_dec ${lr_dec} \
-save_checkpoint_steps 500 \
-batch_size 128 \
-train_steps 20000 \
-report_every 50 \
-accum_count 5 \
-use_bert_emb true \
-use_interval true \
-warmup_steps_bert 2000 \
-warmup_steps_dec 1500 \
-max_pos 512 \
-visible_gpus $gpus \
-log_file ../logs/$log_file3 \
-encoder bert \
-seed 222