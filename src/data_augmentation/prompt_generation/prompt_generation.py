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
from prompt_templates import get_zero_shot_template_tacred, get_zero_shot_template_tacred_rag

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
    rel_conditions = TACRED_VALID_CONDITIONS

    rel_explanations = TACRED_LABEL_TEMPLATES
    # rel_explanations = RETACRED_LABEL_TEMPLATES
    for index, line in enumerate(test_data):
        sel_relations = []

        # sentence = " ".join([ token for token in line['tokens']])
        sentence = " ".join([token for token in line['token']])
        relation = line['relation']
        ss, se = line['subj_start'], line['subj_end']
        os, oe = line['obj_start'], line['obj_end']
        st, ot = line['subj_type'], line['obj_type']
        ents_type = st + ":" + ot
        # head, tail = sentence[ss: se + 1], sentence[os: oe + 1]
        head, tail = line['token'][ss: se + 1], line['token'][os: oe + 1]
        head = " ".join([token for token in head])
        tail = " ".join([token for token in tail])

        for rel in relations:
            if rel in rel_conditions:
                values = rel_conditions[rel]
                if ents_type in values:
                    sel_relations.append(rel)
        sel_relations.append('no_relation')

        sel_explanations = {}
        for relation in sel_relations:
        # for relation in relations:
            values = rel_explanations.get(relation, [])

            if len(values) > 1:
                values = 'or '.join(values)
            else:
                values = values[0]

            explanation = values.format(subj=head, obj=tail)
            sel_explanations[relation] = explanation

        formatted = []
        for key, value in sel_explanations.items():
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
        pass
    else:

        if prompt_type == "simple":
            # prompts = tacred_format(sentences, relations, similar_sentences)
            prompts = tacred_format(train_data, sentences, relations, similar_sentences)
        else:
            # prompts = tacred_format(sentences, relations, similar_sentences, prompt_type)
            prompts = tacred_format(train_data, sentences, relations, similar_sentences, prompt_type)


    return prompts
    


