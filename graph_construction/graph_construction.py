import stanza
import os
import json
from tqdm import tqdm
import sys,os
import pickle
import numpy as np
import sys,os

nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'radiology'})
nlp2 = stanza.Pipeline('en', package='mimic', processors='tokenize')

EMB_INIT_RANGE = 1.0
stop_words = ['XXXX','.',',',';',':']

def get_single_entity_graph(document,impression,entity_modified=True,entity_interval=True,entity_deparser=True):
    doc = nlp(document)
    imp = nlp2(impression)
    fingings_list = []
    impression_list = []
    current_senquence_num = 0
    print(document)
    print(doc)
    edges = []
    edge_words = []
    edges_type = dict()
    edges_type['deparser'] = []
    edges_type['interval'] = []
    graphs = []
    entities = []
    words_id_entities = []

    for sentence in imp.sentences:
        for i in range(len(sentence.words)):
            impression_list.append(sentence.words[i].text)

    for sentence in doc.sentences:
        sent = []
        for i in range(len(sentence.words)):
            sent.append(sentence.words[i].text)
        fingings_list.append(sent)


        entities_sentence = []
        entities_id_sentence = []
        words_entities_sentence = []
        words_id_entities_sentence = []
        entities_type = []
        edges_sentence = []
        edges_word_sentence = []
        entities_type_dict = dict()
        edges_type_sentence = dict()
        edges_type_sentence['deparser'] = []
        edges_type_sentence['interval'] = []


        for i in range(len(sentence.tokens)):


            token = sentence.tokens[i]
            ent_token = token.ner
            text_token = token.text
            id_token = token.id[0]

            if 'UNCERTAINTY' not in ent_token:

                if ent_token!='O':
                    ent_index = ent_token.split('-')[0]
                    words_entities_sentence.append(text_token)
                    words_id_entities_sentence.append(id_token)
                    current_ent_type = ent_token.split('-')[-1]
                    if ent_index == 'S':
                        entities_sentence.append([text_token])
                        entities_id_sentence.append([id_token])
                        entities_type.append(current_ent_type)

                        if current_ent_type not in entities_type_dict:
                            entities_type_dict[current_ent_type] = [id_token]
                        else:
                            entities_type_dict[current_ent_type].append(id_token)

                    elif ent_index == 'B':
                        entities_sentence.append([text_token])
                        entities_id_sentence.append([id_token])

                    elif ent_index == 'I':
                        try:
                            entities_sentence[-1].append(text_token)
                            entities_id_sentence[-1].append(id_token)
                        except:
                            entities_sentence.append([text_token])
                            entities_id_sentence.append([id_token])


                    elif ent_index == 'E':
                        entities_sentence[-1].append(text_token)
                        entities_id_sentence[-1].append(id_token)
                        entities_type.append(ent_token.split('-')[-1])

                        if current_ent_type not in entities_type_dict:
                            entities_type_dict[current_ent_type] = entities_id_sentence[-1]
                        else:
                            entities_type_dict[current_ent_type] = entities_type_dict[current_ent_type]+entities_id_sentence[-1]

        words_id_entities.append(words_id_entities_sentence)
        if entity_deparser:
            if 'deparser' not in edges_type_sentence.keys():
                edges_type_sentence['deparser'] = []


            for word in sentence.words:
                word_id = word.id
                word_head = word.head

                if word_id in words_id_entities_sentence or word_head not in words_id_entities_sentence:
                    word_doc_id = word_id + current_senquence_num - 1
                    word_doc_head = word_head + current_senquence_num - 1

                    if (sentence.words[word_head - 1].text not in stop_words) and \
                            (sentence.words[word_id - 1].text not in stop_words):

                        if word_doc_id >= 0 and word_doc_head >= 0  \
                                            and [word_doc_head,word_doc_id] not in edges_sentence:

                            edges_sentence.append([word_doc_head, word_doc_id])
                            edges_type_sentence['deparser'].append([word_doc_head, word_doc_id])

                            print(sentence.words[word_head - 1].text, sentence.words[word_id - 1].text)

                            edges_word_sentence.append(
                                [sentence.words[word_head - 1].text, sentence.words[word_id - 1].text])


        if entity_interval:
            if 'interval' not in edges_type_sentence.keys():
                edges_type_sentence['interval'] = []

            for m in range(len(entities_id_sentence)):
                entity_length = len(entities_id_sentence[m])
                if entity_length>1:
                    for n in range(entity_length-1):
                        current_id = entities_id_sentence[m][n]
                        current_tag_id = entities_id_sentence[m][n+1]
                        current_doc_id = current_id+current_senquence_num-1
                        current_doc_tag_id = current_tag_id +current_senquence_num-1

                        if (sentence.words[current_tag_id - 1].text not in stop_words) and \
                                (sentence.words[current_id - 1].text not in stop_words):

                            if current_doc_id>=0 and current_doc_tag_id>=0 and [current_doc_id,current_doc_tag_id] not in edges_sentence \
                                   and [current_doc_tag_id, current_doc_id] not in edges_sentence  :

                                    edges_sentence.append([current_doc_id, current_doc_tag_id])
                                    edges_sentence.append([current_doc_tag_id, current_doc_id])
                                    edges_word_sentence.append([sentence.words[current_id-1].text,sentence.words[current_tag_id-1].text])
                                    edges_word_sentence.append([sentence.words[current_tag_id-1].text,sentence.words[current_id-1].text])

                            if current_doc_id>=0 and current_doc_tag_id>=0:
                                edges_type_sentence['interval'].append([current_doc_id, current_doc_tag_id])
                                edges_type_sentence['interval'].append([current_doc_tag_id,current_doc_id])



        current_senquence_num = 0

        edges_type['deparser'].append(edges_type_sentence['deparser'])
        edges_type['interval'].append(edges_type_sentence['interval'])

        edges.append(edges_sentence)
        edge_words.append(edges_word_sentence)
    pyg_edges_document = []



    for edges_single_sentence in edges:
        src_index = []
        tag_index = []
        for edge_item in edges_single_sentence:
            src_index.append(edge_item[0])
            tag_index.append(edge_item[1])
        pyg_edges_document.append([src_index,tag_index])


    return pyg_edges_document,edge_words,fingings_list,impression_list,edges_type,words_id_entities


def build_entity_graph(data_path,entity_interval=True,entity_deparser=True):
    file = open(data_path, 'r', encoding='utf-8')
    lines = file.readlines()
    num_line = len(lines)
    new_json_path = data_path.replace('.jsonl', '')
    name_type = '_real_entity_with_graph'


    new_json_path = new_json_path + name_type + '.jsonl'
    if (os.path.exists(new_json_path)):
        print('there are already exist ' + new_json_path)
        return new_json_path
    else:
        new_json_file = open(new_json_path, 'w', encoding='utf-8')
        for i in tqdm(range(num_line)):
            dic_items = json.loads(lines[i])
            findings_list = dic_items['findings']
            findings = ' '.join(findings_list)
            impression_list = dic_items['impression']
            impression = ' '.join(impression_list)
            edges_with_nodeid = []

            edges,edge_words,fingings_list,impression_list,edges_type_sentence,words_id_entities = get_single_entity_graph(findings,impression,
                                                                                     entity_interval=entity_interval,
                                                                                     entity_deparser=entity_deparser)

            dic_items['pyg_edges_document'] = edges
            dic_items['findings'] = fingings_list
            dic_items['impression'] = impression_list
            dic_items['edge_words'] = edge_words
            dic_items['words_id_entities'] = words_id_entities
            edges = dic_items['pyg_edges_document']






            findings_one_list = []
            for seq in fingings_list:
                findings_one_list = findings_one_list + seq

            if len(findings_one_list)>10 and len(impression_list)>3:
                print(json.dumps(dic_items), file=new_json_file)




if __name__ == '__main__':

    build_entity_graph('./example.jsonl')