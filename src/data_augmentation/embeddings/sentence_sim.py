import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"


import configparser

def read_json(path):
    """ Read json file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    """ Write json file"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        

def compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """
    重写的相似度计算函数
    Compute Consine similarity between test and train embeddings

    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """

    test_similarities = []
    k = 7
    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []
        # similarities = F.cosine_similarity(test_emb.unsqueeze(0), train_embeddings,dim=1)  # SBERT
        similarities = F.cosine_similarity(test_emb.unsqueeze(0), train_embeddings,dim=2)  # UniST
        similarities = similarities.squeeze(1)  # UniST

        # 选前k个
        top_k_indices = torch.argsort(similarities, descending=True)[:k]
        top_k_scores = similarities[top_k_indices]

        top_k_indices = top_k_indices.tolist()
        top_k_scores = top_k_scores.tolist()

        sim_sentence = []
        sim_head = []
        sim_tail = []
        sim_rel = []
        # 读取前k个的信息
        for index, score in zip(top_k_indices, top_k_scores):
            train_emb = train_embeddings[index][0]
            train_sentence = " ".join(train_data[index]['token'])
            context = train_sentence
            ss, se = train_data[index]['subj_start'], train_data[index]['subj_end']
            os, oe = train_data[index]['obj_start'], train_data[index]['obj_end']
            head, tail = train_data[index]['token'][ss: se + 1], train_data[index]['token'][os: oe + 1]  # 根据 两个实体的位置 获取实体name信息 e1:['his'] e2:['him']
            head = " ".join([token for token in head])
            tail = " ".join([token for token in tail])
            sim_sentence.append(context)
            sim_head.append(head)
            sim_tail.append(tail)
            sim_rel.append(train_data[index]['relation'])
            # example.append({"train": index, "simscore": score, "sentence": context})
        # most_sim_index = torch.argmax(similarities)

        # similarities.append({"test": test_index, "similar_sentence": train_similarities[0]['sentence'],
        #                      "train_idex": train_similarities[0]['train'],
        #                      "simscore": float(train_similarities[0]['simscore'])})
        test_similarities.append({"test": test_index, "similar_sentence": sim_sentence,
                             "train_idex": top_k_indices,
                             "simscore": top_k_scores,
                             "similar_head": sim_head,
                             "similar_tail": sim_tail,
                             "similar_rel": sim_rel })

        print("test index: ", test_index)

    return test_similarities

def compute_similarity_new_for_semeval(test_data, train_data, train_embeddings, test_embeddings):
    """
    重写的相似度计算函数
    Compute Consine similarity between test and train embeddings

    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings

    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """

    test_similarities = []
    k = 5
    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []
        similarities = F.cosine_similarity(test_emb.unsqueeze(0), train_embeddings)

        # 选前k个
        top_k_indices = torch.argsort(similarities, descending=True)[:k]
        top_k_scores = similarities[top_k_indices]

        top_k_indices = top_k_indices.tolist()
        top_k_scores = top_k_scores.tolist()

        sim_sentence = []
        sim_head = []
        sim_tail = []
        sim_rel = []
        # 读取前k个的信息
        for index, score in zip(top_k_indices, top_k_scores):
            train_emb = train_embeddings[index]
            train_sentence = " ".join(train_data[index]['tokens'])
            context = train_sentence
            ss, se = train_data[index]['entities'][0][0], train_data[index]['entities'][0][1] - 1
            os, oe = train_data[index]['entities'][1][0], train_data[index]['entities'][1][1] - 1
            head, tail = train_data[index]['tokens'][ss: se + 1], train_data[index]['tokens'][os: oe + 1]  # 根据 两个实体的位置 获取实体name信息 e1:['his'] e2:['him']
            head = " ".join([token for token in head])
            tail = " ".join([token for token in tail])
            sim_sentence.append(context)
            sim_head.append(head)
            sim_tail.append(tail)
            sim_rel.append(train_data[index]['label'])

        test_similarities.append({"test": test_index, "similar_sentence": sim_sentence,
                             "train_idex": top_k_indices,
                             "simscore": top_k_scores,
                             "similar_head": sim_head,
                             "similar_tail": sim_tail,
                             "similar_rel": sim_rel })

        print("test index: ", test_index)

    return test_similarities

def semeval_compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """Compute Consine similarity between test and train embeddings for semeval dataset
    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings
    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """
    
    similarities = []

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):
            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_similarities.append({"train":train_index, "simscore":sim, "sentence":train_line})
        
        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        # similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})
        similarities.append({"test": test_index, "similar_sentence": train_similarities[0]['sentence'], "train_idex": train_similarities[0]['train'], "simscore": float(train_similarities[0]['simscore']),
                             "similar_sentence1": train_similarities[1]['sentence'],
                             "train_idex1": train_similarities[1]['train'],
                             "simscore1": float(train_similarities[1]['simscore'])
                             })

        print("test index: ", test_index)

    return similarities


def main(test_file, train_file, train_emb, test_emb, output_sim_path, dataset="semeval"):
    """Compute similarity between test and train embeddings"""

    test_data = read_json(test_file)
    train_data = read_json(train_file)

    train_embeddings = np.load(train_emb)
    test_embeddings = np.load(test_emb)
    # add 转换成tensor格式
    train_embeddings = torch.Tensor(train_embeddings)
    test_embeddings = torch.Tensor(test_embeddings)

    if dataset == "semeval":
        similarities = compute_similarity_new_for_semeval(test_data, train_data, train_embeddings, test_embeddings)
    else:
        similarities = compute_similarity(test_data, train_data, train_embeddings, test_embeddings)

    write_json(output_sim_path, similarities)


if __name__ == "__main__":

    config = configparser.ConfigParser()
    # config.read(PREFIX_PATH+"config.ini")
    config.read('/home/tianlei/sshCode/RelAware-RAG/src/config/config_tacred.ini')

    test_file = config["SIMILARITY"]["test_file"]
    train_file = config["SIMILARITY"]["train_file"]
    train_emb = config["SIMILARITY"]["train_emb"]
    test_emb = config["SIMILARITY"]["test_emb"]
    # 新增
    dataset = config["SETTINGS"]["dataset"]
    output_sim_path = config["SIMILARITY"]["output_index"]

    # main(test_file, train_file, train_emb, test_emb, output_sim_path)
    main(test_file, train_file, train_emb, test_emb, output_sim_path, dataset)