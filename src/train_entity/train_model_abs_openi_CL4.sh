#

gpus=7
lr_bert=0.0001
lr_dec=0.01

save_interval=500
train_steps=30000
interval_num=61


echo $lr_bert
echo $lr_dec

bert_data_path=../biobert_entity_openi_data/radiology/radiology


for seed in 222 333 444 888 999
do

    Model_path=openi_model_biobert_${seed}_lr_bert_${lr_bert}_lr_dec_${lr_dec}_640
    log_file=$Model_path.log

    echo $Model_path

#     CUDA_VISIBLE_DEVICES=$gpus python train.py  \
#     -task abs \
#     -mode train \
#     -bert_data_path ${bert_data_path} \
#     -dec_dropout 0.2  \
#     -model_path $Model_path \
#     -sep_optim true \
#     -lr_bert ${lr_bert} \
#     -lr_dec ${lr_dec} \
#     -save_checkpoint_steps $save_interval \
#     -batch_size 640 \
#     -train_steps ${train_steps} \
#     -report_every 50 \
#     -accum_count 1 \
#     -use_bert_emb true \
#     -use_interval true \
#     -warmup_steps_bert 2000 \
#     -warmup_steps_dec 1500 \
#     -max_pos 512 \
#     -visible_gpus $gpus \
#     -log_file ../logs/$log_file \
#     -encoder bert \
#     -seed ${seed}

    echo $Model_path
    log_file2=$Model_path.test.log
    result_file=$Model_path.result

    for((i=1;i<=${interval_num};i++));
        do
        let "step = 0+i*$save_interval"
        test_model=model_step_$step.pt
        echo $test_model
        echo $Model_path/$test_model
        CUDA_VISIBLE_DEVICES=$gpus python train.py \
        -task abs \
        -mode test \
        -batch_size 1000 \
        -test_batch_size 500 \
        -bert_data_path ${bert_data_path} \
        -log_file ../logs/$log_file2 \
        -model_path $Model_path \
        -sep_optim true \
        -use_interval true \
        -visible_gpus $gpus \
        -max_pos 512 \
        -max_length 50 \
        -alpha 0.95 \
        -min_length 6 \
        -result_path ../logs/$result_file \
        -test_from $Model_path/$test_model

        rm -r $Model_path/$test_model
    done

done

