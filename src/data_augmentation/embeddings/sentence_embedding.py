""" This script is used to compute the sentence embeddings for the sentences in the dataset."""
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaTokenizer
import torch

from src.data_augmentation.embeddings.data import TACREDDataset
from src.data_augmentation.embeddings.model import UniSTModel

"""Created by: LeiTian"""
import os
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import configparser

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"


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


def compute_sentence(data):
    max_sent_length = 160
    max_label_length = 20
    sent_embeddings = []
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    config = RobertaConfig.from_pretrained("/home/tianlei/work/RelAware-RAG/src/model/tacrev/roberta-large")
    tokenizer = RobertaTokenizer.from_pretrained("/home/tianlei/work/RelAware-RAG/src/model/tacrev/roberta-large")
    model = UniSTModel.from_pretrained("/home/tianlei/work/RelAware-RAG/src/model/tacrev/roberta-large",
                                       config=config)
    model.to(device)
    # add new token
    special_tokens_dict = {
        "additional_special_tokens": ["<E>", "</E>", "<SUBJ>", "</SUBJ>", "<OBJ>", "</OBJ>", "<T>", "</T>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens")
    model.resize_token_embeddings(len(tokenizer))
    # roberta = model.roberta

    print("The embeddings will be compted for {0} sentences".format(len(data)))
    # infer
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for i, line in enumerate(data):
            sent, pos, neg = line[0], line[1], line[2]
            sent_inputs = tokenizer(sent, padding=True, truncation=True, max_length=max_sent_length,
                                    return_tensors="pt", is_split_into_words=True).to(device)
            pos_inputs = tokenizer(pos, padding=True, truncation=True, max_length=max_label_length,
                                   return_tensors="pt").to(device)
            neg_inputs = tokenizer(neg, padding=True, truncation=True, max_length=max_label_length,
                                   return_tensors="pt").to(device)

            inputs = {
                "sent_input_ids": sent_inputs["input_ids"],
                "pos_input_ids": pos_inputs["input_ids"],
                "neg_input_ids": neg_inputs["input_ids"],
                "sent_attention_mask": sent_inputs["attention_mask"],
                "pos_attention_mask": pos_inputs["attention_mask"],
                "neg_attention_mask": neg_inputs["attention_mask"],
            }

            sent_embedding = model(**inputs)[1]

            # # 计算实体嵌入
            # rel_embed = RE(clean_sent)
            sent_embeddings.append(sent_embedding)
            # sent_embeddings.append(embeddings)
            print("Processed sentence: ", i)

    print("The embeddings were completed for {0} sentences".format(len(sent_embeddings)))

    return sent_embeddings


def write_embeddings(embeddings, output_file):
    numpy_embeddings = [embedding.detach().cpu().numpy() for embedding in embeddings]
    numpy_embeddings = np.stack(numpy_embeddings)
    np.save(output_file, numpy_embeddings)

if __name__ == "__main__":

    print("PREFIX_PATH", PREFIX_PATH)

    file_config = configparser.ConfigParser()
    # config.read(PREFIX_PATH+"config.ini")
    file_config.read("/home/tianlei/work/RelAware-RAG/src/config/config_re_tacred.ini")

    input_file = file_config["EMBEDDING"]["input_embedding_path"]
    output_file = file_config["EMBEDDING"]["output_embedding_path"]

    # tacred
    tacred_train_dataset = TACREDDataset(input_file)

    embeddings = compute_sentence(tacred_train_dataset)  # idea1:修改embedding计算方法
    write_embeddings(embeddings, output_file)