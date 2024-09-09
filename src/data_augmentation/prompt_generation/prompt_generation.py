import os
import sys
import json

import re
import random

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.template.templates import *

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
from prompt_templates import get_zero_shot_template_tacred, get_zero_shot_template_tacred_rag, semeval_prompt_template_rag, semeval_prompt_template

def read_json(path):
    """Read json file"""

    with open(path, 'r') as f:
        data = json.load(f)

    return data

def tacred_format(train_data, test_data, relations, similar_sentences, type="rag"):
    """Regenerate prompt for tacred and its variants like tacrev, re-tacred

    Args:
        test_data (list): list of test data
        relations (list): list of relations (target labels)
        similar_sentences (list): list of similar sentence with corresponding test data
        type (str, optional): prompt type. Defaults to "rag".

    Returns:
        list: list of regenerated prompts
    """
    
    prompts = []

    # relations = " ".join([relation for relation in relations])
    # all_relations = " ".join([relation for relation in relations])
    rel_conditions = TACRED_VALID_CONDITIONS  # 筛选的条件
    # rel_conditions = RETACRED_VALID_CONDITIONS  # 筛选的条件


    # 添加了 限定关系类型、关系类型解释说明
    rel_explanations = TACRED_LABEL_TEMPLATES
    # rel_explanations = RETACRED_LABEL_TEMPLATES
    for index, line in enumerate(test_data):
        sel_relations = []  # 筛选的结果

        # sentence = " ".join([ token for token in line['tokens']])
        sentence = " ".join([token for token in line['token']])
        relation = line['relation']  # 获取关系label relation:'no_relation'
        ss, se = line['subj_start'], line['subj_end']
        os, oe = line['obj_start'], line['obj_end']
        st, ot = line['subj_type'], line['obj_type']
        ents_type = st + ":" + ot
        # head, tail = sentence[ss: se + 1], sentence[os: oe + 1]  # 根据 两个实体的位置 获取实体name信息 e1:['his'] e2:['him']
        head, tail = line['token'][ss: se + 1], line['token'][os: oe + 1]  # 根据 两个实体的位置 获取实体name信息 e1:['his'] e2:['him']
        head = " ".join([token for token in head])
        tail = " ".join([token for token in tail])

        for rel in relations:
            # 检查 rel 是否在 rel_conditions 的键中
            if rel in rel_conditions:
                values = rel_conditions[rel]
                if ents_type in values:
                    sel_relations.append(rel)
        sel_relations.append('no_relation')

        # 获取对应的关系解释
        sel_explanations = {}
        for relation in sel_relations:
        # for relation in relations:
            # 获取当前关系类型的值
            values = rel_explanations.get(relation, [])

            # 如果值是多个字符串数组，将其转换为一个由 or 分隔的字符串
            if len(values) > 1:
                values = 'or '.join(values)
            else:
                values = values[0]

            # 替换subj和obj，然后添加到sel_relations_explain字典中
            explanation = values.format(subj=head, obj=tail)
            sel_explanations[relation] = explanation

        formatted = []
        for key, value in sel_explanations.items():
            # 使用format方法将键和值格式化为字符串
            tmp = "{}: {}".format(key, value)
            formatted.append(tmp)
        sel_explanations_formatted = ',  '.join(formatted)

        sel_relations = " ".join([rel for rel in sel_relations])

        if type == "simple":
            # prompt = get_zero_shot_template_tacred(sentence, relations, head, tail)
            prompt = get_zero_shot_template_tacred(sentence, sel_relations, head, tail)
            data = {"prompt": prompt, "relation": line['relation']}
        else:
            context = similar_sentences[index]
            similar_sentence = context['similar_sentence']
            # 增加相似train_set中的实体对信息
            similar_train_head = context['similar_head']
            similar_train_tail = context['similar_tail']
            similar_train_relation = context['similar_rel']
            train_contexts = []
            for sim_sen, sim_head, sim_tail, sim_rel in zip(similar_sentence, similar_train_head, similar_train_tail,
                                                            similar_train_relation):
                train_context = {"similar_sentence": sim_sen, "similar_train_head": sim_head,
                                 "similar_train_tail": sim_tail, "similar_train_relation": sim_rel}
                train_contexts.append(train_context)

            valid_relations = {"sel_relations": sel_relations, "sel_explanations_formatted": sel_explanations_formatted}
            # valid_relations = {"sel_relations": all_relations, "sel_explanations_formatted": sel_explanations_formatted}
            # prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, context['similar_sentence'])
            # prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, train_contexts)
            prompt = get_zero_shot_template_tacred_rag(sentence, valid_relations, head, tail, train_contexts)
            data = {"prompt": prompt, "relation": line['relation']}
        prompts.append(data)
    print("Number Prompts:{0}".format(len(prompts)))

    return prompts



def semeval_format(test_data, relations, similar_sentences, type="simple"):
    """Regenerate prompt for semeval dataset

    Args:
        test_data (list): list of test sentences along with e1 and e2
        relations (list): target relation label indexes
        similar_sentences (list): list of similar sentence with corresponding test data
        labels (list): the list  of target label names
        prompt_type (str, optional): prompt type. Defaults to "simple".

    Returns:
        list: the list of regenerated prompts
    """
    
    # relation_names = list(set(relations))
    # relations = ", ".join([relation for relation in relation_names])
    # prompts = []
    prompts = []
    # 改
    # relations = " ".join([relation for relation in relations])

    # for index, line in enumerate(test_data):
    #
    #     label = labels[index]
    #     sentence = line
    #     context = similar_sentences[index]
    #
    #     e1_index = sentence.find("<e1>")
    #     e2_index = sentence.find("<e2>")
    #
    #     if e1_index < e2_index:
    #         head_name = re.findall("<e1>(.*?)</e1>", sentence, re.DOTALL)
    #         tail_name = re.findall("<e2>(.*?)</e2>", sentence, re.DOTALL)
    #         head = "e1"
    #         tail = "e2"
    #     else:
    #         # print("e2")
    #         head_name = re.findall("<e2>(.*?)</e2>", sentence, re.DOTALL)
    #         tail_name = re.findall("<e1>(.*?)</e1>", sentence, re.DOTALL)
    #         head = "e2"
    #         tail = "e1"
    #
    #     head_name = " ".join(head_name)
    #     tail_name = " ".join(tail_name)
    #
    #     if prompt_type == "simple":
    #         prompt = semeval_prompt_template(sentence, relations, head, tail, head_name, tail_name)
    #
    #     if prompt_type == "rag":
    #         context = context[index]
    #         prompt = semeval_prompt_template_rag(sentence, relations, head, tail, head_name, tail_name, context['similar_sentence'])
    #
    #     prompts.append({"prompt":prompt, "relation":label})
    #
    # print("Number of Prompts:{0}".format(len(prompts)))

    # 添加了 限定关系类型、关系类型解释说明
    # rel_conditions = TACRED_VALID_CONDITIONS  # 筛选的条件
    rel_explanations = SEMEVAL_LABEL_TEMPLATES

    for index, line in enumerate(test_data):
        sentence = " ".join([token for token in line['tokens']])
        relation = line['label']  # 获取关系label relation:'no_relation'
        ss, se = line['entities'][0][0], line['entities'][0][1]
        os, oe = line['entities'][1][0], line['entities'][1][1]

        head, tail = line['tokens'][ss: se], line['tokens'][os: oe]  # 根据 两个实体的位置 获取实体name信息 e1:['his'] e2:['him']
        head = " ".join([token for token in head])
        tail = " ".join([token for token in tail])

        # sel_explanations = {
        #     relation_type: template[0].format(subj=head, obj=tail)
        #     for relation_type, template in rel_explanations.items()
        # }
        #
        # formatted = []
        # for key, value in sel_explanations.items():
        #     # 使用format方法将键和值格式化为字符串
        #     tmp = "{}: {}".format(key, value)
        #     formatted.append(tmp)
        sel_explanations = [
            template[0].format(subj=head, obj=tail)
            for relation_type, template in rel_explanations.items()
        ]

        new_formatted = []
        for idx, item in enumerate(sel_explanations, start=1):
            formatted_item = f"[{idx}] {item}\n"
            new_formatted.append(formatted_item)

        sel_explanations_formatted = ''.join(new_formatted)

        sel_relations = " ".join([rel for rel in relations])

        if type == "simple":
            # prompt = get_zero_shot_template_tacred(sentence, relations, head, tail)
            prompt = get_zero_shot_template_tacred(sentence, sel_relations, head, tail)
            data = {"prompt": prompt, "relation": line['relation']}
        else:
            context = similar_sentences[index]
            similar_sentence = context['similar_sentence']
            # 增加相似train_set中的实体对信息
            similar_train_head = context['similar_head']
            similar_train_tail = context['similar_tail']
            similar_train_relation = context['similar_rel']
            train_contexts = []
            for sim_sen, sim_head, sim_tail, sim_rel in zip(similar_sentence, similar_train_head, similar_train_tail,
                                                            similar_train_relation):
                train_context = {"similar_sentence": sim_sen, "similar_train_head": sim_head,
                                 "similar_train_tail": sim_tail, "similar_train_relation": sim_rel}
                train_contexts.append(train_context)

            valid_relations = {"sel_relations": sel_relations, "sel_explanations_formatted": sel_explanations_formatted}
            # prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, context['similar_sentence'])
            # prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, train_contexts)
            prompt = get_zero_shot_template_tacred_rag(sentence, valid_relations, head, tail, train_contexts)
            data = {"prompt": prompt, "relation": line['label']}
        prompts.append(data)
    print("Number Prompts:{0}".format(len(prompts)))

    return prompts
  

# def generate_prompts(sentences, relations, similar_sentences,  dataset="tacred", prompt_type="rag"):
def generate_prompts(train_data, sentences, relations, similar_sentences,  dataset="tacred", prompt_type="rag"):
    """Regenerate the user query along with similar sentence.

    Args:
        sentences (list): list of sentences or dataset
        relations (list): list of relations
        similar_sentences (list): list of similar sentences
        dataset (str, optional): dataset name. Defaults to "tacred".
        prompt_type (str, optional): approach type. Defaults to "rag".

    Returns:
        list of prompts: list of regenerated prompts
    """

    prompts = []

    if dataset == "semeval":

        if type == "simple":
            # prompts = semeval_format(sentences, relations, relations)
            prompts = semeval_format(sentences, relations, similar_sentences)
        else:
            # prompts = semeval_format(sentences, relations, relations, prompt_type)
            prompts = semeval_format(sentences, relations, similar_sentences, prompt_type)
    else:

        if prompt_type == "simple":
            # prompts = tacred_format(sentences, relations, similar_sentences)
            prompts = tacred_format(train_data, sentences, relations, similar_sentences)
        else:
            # prompts = tacred_format(sentences, relations, similar_sentences, prompt_type)
            prompts = tacred_format(train_data, sentences, relations, similar_sentences, prompt_type)


    return prompts
    


