# RelAwareRAG

The repository consists of the source codes of "Enhancing Relation Extraction by Relation Awareness-Based Retrieval Augmented Generation" paper.

Note that TACRED is licensed by the Linguistic Data Consortium (LDC), so we cannot directly publish the prompts or the raw results from the experiments conducted with Llamal, since the responses of these models consists of the prompts in their instruction parts.  Upon an official request, the data can be accessed on LDC, and the experiments can be easily replicated by following the instructions provided.

## Project Folder Hierarchy

````
.
├── README.md
├── data                            ---> dataset, such as tacred, tacrev and re-tacred
├── results                         ---> results will be saved here.
└── src
    ├── config.ini                  ---> configuration for dataset, approach and llm and results.
    ├── data_augmentation           ---> regenerated the test input
    ├── main.py                     ---> the pipeline is started with this
    ├── retrieval                   ---> retrieval module
    ├── generation_module           ---> llm prompting.
    ├── evaluation                  ---> evaluate and visualize results. 
    ├── template                  ---> template Te and Tr
    └── utils.py                    
````
## How to run
Change the paths and configs under `config.ini` for your experiment.
* 1.) Datasets
  
   Put the following dataset under `data` folder.
  
   * TACRED dataset is lincensed by Linguistic Data Consortium (LDC), so please download it from [here](https://catalog.ldc.upenn.edu/LDC2018T24)
   * TACREV dataset is constructed from TACRED via the tacrev [codes](https://github.com/DFKI-NLP/tacrev)
   * Re-TACRED dataset is derived from TACRED via this [repository](https://github.com/gstoica27/Re-TACRED)
   
* 2.) First install requirements
* 3.) Compute embeddings and similarities for benchmark datasets in advance
````bash
    cd src/data_augmentation/embeddings
    python sentence_embeddings.py
    python sentence_sim.py
````
* 4.) Run Project
````bash
$ python src/main.py
````

