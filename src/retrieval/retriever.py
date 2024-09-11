import os
import sys
import configparser

from tqdm import tqdm

# from data_augmentation.prompt_generation.prompt_generation import generate_prompts
from src.data_augmentation.prompt_generation.prompt_generation import generate_prompts
from src.generation_module.generation import LLM
import configparser

from src.utils import read_json, write_json
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
from refinement import postprocessing

def benchmark_data_augmentation_call(config_file_path):
    """
    This function is used to benchmark the retrieval module.
    Args:
    config_file_path: str: Path to the config file.
    """
    # 新增
    print("进入benchmark_data_augmentation_call")

    print("PREFIX_PATH", PREFIX_PATH)

    config = configparser.ConfigParser()
    config.read("/home/tianlei/sshCode/RelAware-RAG/src/config/config.ini")

    test_data_path = config["PATH"]["test_data_path"]
    #  增加train set，为了读取训练集信息
    train_data_path = config["PATH"]["train_data_path"]
    similar_sentences_path = config["SIMILARITY"]["output_index"]
    relations_path = config["PATH"]["relations_path"]

    # The training set sentences in the vector database that are similar to the test input sentences
    similar_sentences = read_json(similar_sentences_path)
    relations = read_json(relations_path)
    relations = relations.keys()
    test_data = read_json(test_data_path)
    # add train data
    train_data = read_json(train_data_path)

    dataset = config["SETTINGS"]["dataset"]  # tacred
    prompt_type = config["SETTINGS"]["prompt_type"]
    model_name = config["SETTINGS"]["model_name"]

    if prompt_type == "rag":
        output_prompts_path = config["OUTPUT"]["rag_test_prompts_path"]
        output_responses_path = config["OUTPUT"]["rag_test_responses_path"]
        prompts = generate_prompts(train_data, test_data, relations, similar_sentences,  dataset, prompt_type)
    elif prompt_type == "random":
        output_prompts_path = config["OUTPUT"]["rag_test_prompts_path"]
        output_responses_path = config["OUTPUT"]["rag_test_responses_path"]
        prompts = generate_prompts(train_data, test_data, relations, similar_sentences, dataset, prompt_type)
    else:
        output_prompts_path = config["OUTPUT"]["simple_prompt_path"]
        output_responses_path = config["OUTPUT"]["simple_prompt_responses_path"]
        prompts = generate_prompts(train_data, test_data, relations, similar_sentences,  dataset, prompt_type)

    llm_instance = LLM(model_name)
    
    responses = []

    # for prompt in prompts:
    for prompt in tqdm(prompts, desc="正在llm预测"):
        prompt = prompt["prompt"]

        if not "t5" in model_name:
            # prompt = """[INST]{prompt}[/INST] Answer:"""
            prompt = """[INST]""" + prompt + """[/INST] Answer:"""

        response = llm_instance.get_prediction(prompt)
        responses.append(response)
        # responses.append(response[0])  # Llama大模型

    responses = postprocessing(dataset, test_data, responses, relations, model_name)


    print("Write the output to a json file")
    write_json(output_prompts_path, prompts)
    write_json(output_responses_path, responses)
    
