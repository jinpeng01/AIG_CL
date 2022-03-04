import gc
import glob
import hashlib
import itertools
import json
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin
from tqdm import tqdm

import torch
# from multiprocess import Pool
import stanza

from others.logging import logger
from others.tokenization import BertTokenizer
# from pytorch_transformers import XLNetTokenizer
from transformers import AutoTokenizer
from others.utils import clean
from prepro.utils import _get_word_ngrams

import xml.etree.ElementTree as ET
# nlp = stanza.Pipeline('en', package='mimic', processors='tokenize')
# nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'radiology'})

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]



def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)



def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt



def load_xml(p):
    tree = ET.parse(p)
    root = tree.getroot()
    title, byline, abs, paras = [], [], [], []
    title_node = list(root.iter('hedline'))
    if (len(title_node) > 0):
        try:
            title = [p.text.lower().split() for p in list(title_node[0].iter('hl1'))][0]
        except:
            print(p)

    else:
        return None, None
    byline_node = list(root.iter('byline'))
    byline_node = [n for n in byline_node if n.attrib['class'] == 'normalized_byline']
    if (len(byline_node) > 0):
        byline = byline_node[0].text.lower().split()
    abs_node = list(root.iter('abstract'))
    if (len(abs_node) > 0):
        try:
            abs = [p.text.lower().split() for p in list(abs_node[0].iter('p'))][0]
        except:
            print(p)

    else:
        return None, None
    abs = ' '.join(abs).split(';')
    abs[-1] = abs[-1].replace('(m)', '')
    abs[-1] = abs[-1].replace('(s)', '')

    for ww in nyt_remove_words:
        abs[-1] = abs[-1].replace('(' + ww + ')', '')
    abs = [p.split() for p in abs]
    abs = [p for p in abs if len(p) > 2]

    for doc_node in root.iter('block'):
        att = doc_node.get('class')
        # if(att == 'abstract'):
        #     abs = [p.text for p in list(f.iter('p'))]
        if (att == 'full_text'):
            paras = [p.text.lower().split() for p in list(doc_node.iter('p'))]
            break
    if (len(paras) > 0):
        if (len(byline) > 0):
            paras = [title + ['[unused3]'] + byline + ['[unused4]']] + paras
        else:
            paras = [title + ['[unused3]']] + paras

        return paras, abs
    else:
        return None, None


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1', do_lower_case=False)
        # self.tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1',do_lower_case=False)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused99]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'

        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False,edge_words=None,pyg_edges_document=None,words_id_entities=None):


        xxx =src
        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        # idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        # src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        # sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        text =' '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)

        temp_subtokens = []
        position = 1
        edges_subtoken = [[],[]]
        edges_subtoken_text = [[],[]]
        entities_findings = []

        for send_id in range(len(src)):

            sent = src[send_id]
            entities_sentence = words_id_entities[send_id]
            index_subtoken = dict()

            for word_ind in range(len(sent)):
                temp_word = sent[word_ind]
                token_s = self.tokenizer.tokenize(temp_word)
                index_subtoken[word_ind] = [position + pos for pos in range(len(token_s))]
                position = position + len(token_s)
                temp_subtokens = temp_subtokens + token_s

            sentence_edge = pyg_edges_document[send_id]
            src_indexs = sentence_edge[0]
            tag_indexs = sentence_edge[1]
            assert len(src_indexs) == len(tag_indexs)

            sentence_edge_subtoken_edges_src = []
            sentence_edge_subtoken_edges_tgt = []
            sentence_edges_subtoken_text_src = []
            sentence_edges_subtoken_text_tgt = []
            sentence_entities_subtoken_id = []

            # for src_index, tag_index in zip(src_indexs, tag_indexs):
            #     for src_subtoken_index in index_subtoken[src_index]:
            #         for tgt_subtoken_index in index_subtoken[tag_index]:
            #             sentence_edge_subtoken_edges_src.append(src_subtoken_index)
            #             sentence_edge_subtoken_edges_tgt.append(tgt_subtoken_index)
            #
            #             sentence_edges_subtoken_text_src.append(temp_subtokens[src_subtoken_index - 1])
            #             sentence_edges_subtoken_text_tgt.append(temp_subtokens[tgt_subtoken_index - 1])

            try:
                for src_index,tag_index in zip(src_indexs,tag_indexs):
                    for src_subtoken_index in index_subtoken[src_index]:
                        for tgt_subtoken_index in index_subtoken[tag_index]:
                            sentence_edge_subtoken_edges_src.append(src_subtoken_index)
                            sentence_edge_subtoken_edges_tgt.append(tgt_subtoken_index)

                            sentence_edges_subtoken_text_src.append(temp_subtokens[src_subtoken_index-1])
                            sentence_edges_subtoken_text_tgt.append(temp_subtokens[tgt_subtoken_index-1])


            except:
                import pdb
                pdb.set_trace()

            for entity_index in entities_sentence:
                for entity_subtoken_id in index_subtoken[entity_index-1]:
                    sentence_entities_subtoken_id.append(entity_subtoken_id)

            edges_subtoken[0] = edges_subtoken[0]+sentence_edge_subtoken_edges_src
            edges_subtoken[1] = edges_subtoken[1]+sentence_edge_subtoken_edges_tgt
            edges_subtoken_text[0] = edges_subtoken_text[0] + sentence_edges_subtoken_text_src
            edges_subtoken_text[1] = edges_subtoken_text[1] + sentence_edges_subtoken_text_tgt
            entities_findings = entities_findings + sentence_entities_subtoken_id


        assert temp_subtokens == src_subtokens
            # import pdb
            # pdb.set_trace()

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused99] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        temp_edge_words = []
        for temp in edge_words:
            temp_edge_words = temp_edge_words + temp
        edge_words = temp_edge_words

        nodes_set = set()
        subtoken_edges = []
        for i in range(len(edge_words)):
            src_word = edge_words[i][0]
            src_subtokens = self.tokenizer.tokenize(src_word)
            tag_word = edge_words[i][1]
            tag_subtokens = self.tokenizer.tokenize(tag_word)
            for src_subtoken in src_subtokens:

                for tag_subtoken in tag_subtokens:
                    nodes_set.add(tag_subtoken)
                    nodes_set.add(src_subtoken)
                    if ([src_subtoken, tag_subtoken] not in subtoken_edges):
                        subtoken_edges.append([src_subtoken, tag_subtoken])
        find_vocab = []
        w2i = dict()
        i2w = dict()
        node_id = []
        for subtoken in nodes_set:
            w2i[subtoken] = len(w2i)
            i2w[len(i2w)] = subtoken
            find_vocab.append(subtoken)
            node_id = node_id + self.tokenizer.convert_tokens_to_ids([subtoken])

        edges = []
        src_edge_index = []
        tag_edge_index = []

        for edge in subtoken_edges:
            src_edge_index.append(w2i[edge[0]])
            tag_edge_index.append(w2i[edge[1]])
        edges.append(src_edge_index)
        edges.append(tag_edge_index)


        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        # src_txt = [original_src_txt[i] for i in idxs]
        src_txt = original_src_txt
        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, \
               tgt_txt,edges_subtoken,edges_subtoken_text,edge_words,node_id,edges,entities_findings


def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']

    path = dict()


    path['train'] = '../graph_construction/mimir_random_1/random_train_1.jsonl'
    path['test'] = '../graph_construction/mimir_random_1/random_test_1.jsonl'
    path['valid'] = '../graph_construction/mimir_random_1/random_valid_1.jsonl'

    for corpus_type in datasets:
        a_lst = []
        json_f = path[corpus_type]
        real_name = json_f.split('/')[-1]
        a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, 'radiology.'+real_name.replace('jsonl', '0.bert.pt').replace('_with_entity_modified_interval_deparser',''))))
        print(a_lst)

        if a_lst != []:
            for i in range(len(a_lst)):
                _format_to_bert(a_lst[i])




def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)
    logger.info('Processing %s' % json_file)

    datasets = []
    count_i = 1
    for line in tqdm(open(json_file,'r').readlines()):
        d = json.loads(line)

        # findings = ' '.join(d['findings'])
        # doc = nlp(findings)

        tgt = []
        tgt.append(d['impression'])
        source = d['findings']
        pyg_edges_document = d['pyg_edges_document']
        words_id_entities = d['words_id_entities']

        # for sentence in doc.sentences:
        #     # import pdb
        #     # pdb.set_trace()
        #     sent = []
        #     for i in range(len(sentence.words)):
        #         sent.append(sentence.words[i].text)
        #     source.append(sent)


        # node_words = d['nodes']
        # edges = d['edges_with_nodeid']
        # edge_words = d['edge_words']
        if args.type in d.keys():
            edge_words = d[args.type]
        elif '@@' in args.type:
            types = args.type.split('@@')
            edge_words = d[types[0]]+d[types[1]]
        # import pdb
        # pdb.set_trace()
        assert len(pyg_edges_document) == len(edge_words)
        assert len(pyg_edges_document) == len(source)

        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
        # import pdb
        # pdb.set_trace()
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test,edge_words=edge_words,pyg_edges_document=pyg_edges_document,words_id_entities=words_id_entities)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            continue

        # src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, \
        # src_txt, tgt_txt, node_id, edges,find_vocab,subtoken_edges = b_data
        # import pdb
        # pdb.set_trace()

        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, \
        tgt_txt, edges_subtoken, edges_subtoken_text, edge_words, node_id, edges,\
        entities_findings = b_data

        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt,'edges':edges_subtoken,'node_id':node_id,
                       'subtoken_edges':edges_subtoken_text,'edge_words':edge_words,'edges_node':edges,
                       'entities_findings':entities_findings}

        datasets.append(b_data_dict)
        if(len(datasets) == 2000):
            torch.save(datasets, os.path.join(args.save_path,'radiology.'+corpus_type+'.' + str(count_i) + '.bert.pt'))
            count_i = count_i + 1
            datasets = []

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_lines(args):
    # corpus_mapping = {}
    # for corpus_type in ['valid', 'test', 'train']:
    #     temp = []
    #     for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
    #         temp.append(hashhex(line.strip()))
    #     corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    # train_files, valid_files, test_files = [], [], []
    # for f in glob.glob(pjoin(args.raw_path, '*.json')):
    #     real_name = f.split('/')[-1].split('.')[0]
    #     if (real_name in corpus_mapping['valid']):
    #         valid_files.append(f)
    #     elif (real_name in corpus_mapping['test']):
    #         test_files.append(f)
    #     elif (real_name in corpus_mapping['train']):
    #         train_files.append(f)
        # else:
        #     train_files.append(f)
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        # import pdb
        # pdb.set_trace()
        real_name = f.split('/')[-1].split('.')[0]
        if ('validate' in real_name):
            valid_files.append(f)
        elif ('test' in real_name):
            test_files.append(f)
        elif ('train' in real_name):
            train_files.append(f)
    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []


def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}



