Model_path=openi_model_biobert

gpus=5,6,7
log_file=$Model_path.log

# lr_bert=0.0005
# lr_dec=0.05

lr_bert=0.0003
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
-train_steps 30000 \
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


