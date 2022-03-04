CUDA_VISIBLE_DEVICES=0 python preprocess.py \
-mode format_to_bert \
-raw_path path to_data after constructing graph \
-save_path path to save data  \
-n_cpus 1 \
-log_file ../logs/preprocess.log \
-type edge_words



