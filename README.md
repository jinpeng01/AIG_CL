
# Graph Enhanced Contrastive Learning for Radiology Findings Summarization 
Code for [Graph Enhanced Contrastive Learning for Radiology Findings Summarization](https://aclanthology.org/2022.acl-long.320/)

==========

This repo contains the PyTorch code following [this code](https://github.com/nlpyang/PreSumm)

## Citations

If you use or extend our work, please cite our paper at ACL-2022.
```
@inproceedings{hu-etal-2022-graph,
    title = "Graph Enhanced Contrastive Learning for Radiology Findings Summarization",
    author = "Hu, Jinpeng  and
      Li, Zhuo  and
      Chen, Zhihong  and
      Li, Zhen  and
      Wan, Xiang  and
      Chang, Tsung-Hui",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022"
}
```



## Requirements

- Python 3 (tested on 3.7)
- PyTorch (tested on 1.5)
- We run our experiments on three 32GB-V100
## Data

We give an example about the data in the `graph_construction/`

### Preparation
Remain to be origanized. Some of the code needs to be debug, plz use it carefully.

#### Graph Construction
We have given the example about the data format to construct the graph (each line is a radiology report).
You might need to change the data path to you own data path.
```
cd graph_construction
python graph_construction.py
``` 

After finish graph construction. need to run `sh precess_radiology.sh` to further process data. For this step, you can obtain more information from the link (https://github.com/nlpyang/PreSumm). Note that you also need to change the 322-324 row in src/prepro/data_builder.py to your own data.


## Training
change DATA_PATH to your data_path,
To start training, run

```
sh train_model_abs_openi_CL.sh
```

## Evaluation
change DATA_PATH Model_path to your data_path and model path and let the step to a specific number
To start evaluation, run
```
sh test_openi.sh
```

## Pre-trained model
you can download the pre-trained models from ([the link](https://pan.baidu.com/s/1vE7EIdQF3x-5PgWXpEXrsg)  passwd: co14).
