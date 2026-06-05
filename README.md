# EVT-Rank: Extreme-Value Tail Weighting fo Long-Tail Retrieval-Augmented Generation
EVT-Rank is a method of reranking RAG documents based on the mathematical and physical model of extreme value theory. Compared with the method of sorting documents using LLM, EVT-Rank has lower reordering delay, and can be deployed on the CPU side to avoid possible video memory overflow on GPU side with limited video memory, and has better deployability.

## Setup

### Requirements
EVT-Rank is developed based on the Python platform. The following table lists the environment configuration required by EVT Rank.

| Lib | Version |
| :--------: | :------: |
| PyTorch   | 2.10.0+cu12.8 |
| Transformers | 5.6.2 |
| Pandas	 | 2.3.3 |
| SciPy	 | 1.15.3 |
| NumPy	 | 2.2.6 |

## Datasets
The experiment used the data set of MIRAGE medical RAG benchmark, including 7663 questions from 5 medical QA data sets. MIRAGE is specifically used to evaluate the RAG system, including three medical examination QA data sets (MMLU-Med, MedQAUS, MedMCQA), and two biomedical research QA data sets (PubMedQA*, BioASQ-Y/N). The data set can be obtained at the following website: https://github.com/gzxiong/MIRAGE/tree/main.

In addition, we also selected the general domain dataset MMLU. It covers 57 disciplines and aims to comprehensively evaluate the knowledge breadth and reasoning ability of the model in zero-shot or few-shot scenarios through a wide range of topics ranging from high school level to doctoral level. The download address of the MMLU dataset is https://modelscope.cn/datasets/cais/mmlu/files.

## Retriever
We use the Contriever model as the retriever of the RAG system to vectorize the text. The vector dimension is 768. 
The Contriever retrieval model can be obtained from the following link: https://modelscope.cn/models/facebook/contriever/files.

## Black Box LLM 
For the black-box LLM in our RAG system, we choose Qwen3-0.6B. It is a super lightweight big language model. Although it has only 0.6B parameters, its capabilities are very comprehensive. It is an excellent choice for end-to-end deployment and immediate response.
Qwen3-0.6B model can be downloaded from the following website: https://modelscope.cn/models/Qwen/Qwen3-0.6B/files .

## Corpus for Retriever
We use MedCorp as the external document library of the RAG system, where the original text data comes from four different sources, including the biomedical abstract text set PubMed, the clinical decision support text set StatPearls, the medical textbook text set Textbooks of domain specific knowledge, and the general knowledge text set Wikipedia.
MedCorp is available at: https://github.com/gzxiong/MedRAG.


