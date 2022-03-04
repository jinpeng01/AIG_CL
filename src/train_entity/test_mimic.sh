Model_path=mimic_model_biobert_999


echo $Model_path
log_file2=$Model_path.test.log
result_file=$Model_path.result
bert_data_path=../biobert_entity_mimic_data/radiology/radiology

gpus=0
# input to one checkpoint index
for((i=1;i<=100;i++));
    do
    let "step = 500+i*500"
#     let "step = 80000"
    test_model=model_step_$step.pt
    echo $test_model
    CUDA_VISIBLE_DEVICES=0 python train.py \
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
done

