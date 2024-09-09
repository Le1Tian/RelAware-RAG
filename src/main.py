from retrieval.retriever import benchmark_data_augmentation_call

def main():
    config_file_path = "config_re_tacred.ini"
    benchmark_data_augmentation_call(config_file_path)
if __name__ == "__main__":
    main()