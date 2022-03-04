import os
import re
import shutil
import time

from others import pyrouge
from pythonrouge.pythonrouge import Pythonrouge
REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def process(params):
    temp_dir, data = params
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_recall"] * 100

        # ,results_dict["rouge_su*_f_score"] * 100
    )

def get_rouge(hypotheses, reference, sent_split=True, use_cf=False):
    assert len(hypotheses) == len(reference)
    assert len(hypotheses) > 0

    hyps = []
    refs = []
    # prepare
    for hyp, ref in zip(hypotheses, reference):
        hyp = " ".join(hyp)
        ref = " ".join(ref)
        if sent_split:
            hs = [x.strip() for x in hyp.split('.') if len(x.strip()) > 0]
            rs = [x.strip() for x in ref.split('.') if len(x.strip()) > 0]
            hyps += [hs]
            refs += [[rs]]
        else:
            hyps += [[hyp]]
            refs += [[[ref]]]
    print("Calculating ROUGE...")
    rouge = Pythonrouge(summary_file_exist=False, summary=hyps, reference=refs, \
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True, recall_only=False, stemming=False, stopwords=False, \
                        word_level=True, length_limit=False, use_cf=use_cf, cf=95, scoring_formula='average', \
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print("ROUGE done.")

    r1 = score['ROUGE-1-F'] * 100
    r2 = score['ROUGE-2-F'] * 100
    rl = score['ROUGE-L-F'] * 100
    if not use_cf:

        return r1, r2, rl
        # return results_dict2
    else:
        r1_cf = [x * 100 for x in score['ROUGE-1-F-cf95']]
        r2_cf = [x * 100 for x in score['ROUGE-2-F-cf95']]
        rl_cf = [x * 100 for x in score['ROUGE-L-F-cf95']]
        return r1, r2, rl, r1_cf, r2_cf, rl_cf


def rouge_results_to_str2(results_dict2):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict2["rouge_1_f_score"],
        results_dict2["rouge_2_f_score"],
        results_dict2["rouge_l_f_score"],
    )

def calculate_rouge(predict_path,real_path):
    predictions = open(predict_path,'r').readlines()
    # import pdb
    # pdb.set_trace()
    predictions = [item.replace('<q>','')  for item in predictions]

    predictions = [item.strip().split()  for item in predictions]
    # len(predictions[201])
    # type(predictions[200])
    # import pdb
    # pdb.set_trace()

    truths = open(real_path, 'r').readlines()
    truths = [item.strip().split() for item in truths]
    r1, r2, rl, r1_cf, r2_cf, rl_cf = get_rouge(predictions, truths, use_cf=True)

    print("{} set results:\n".format('test'))
    print("Metric\tScore\t95% CI")
    print("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1, r1_cf[0]-r1, r1_cf[1]-r1))
    print("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2, r2_cf[0]-r2, r2_cf[1]-r2))
    print("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl, rl_cf[0]-rl, rl_cf[1]-rl))

    results_dict2 = dict()
    results_dict2["rouge_1_f_score"] = r1
    results_dict2["rouge_2_f_score"] = r2
    results_dict2["rouge_l_f_score"] = rl
    return results_dict2
