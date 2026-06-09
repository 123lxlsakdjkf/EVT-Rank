# EVT-Rank: Extreme-Value Tail Weighting fo Long-Tail Retrieval-Augmented Generation
EVT-Rank is a method of reranking RAG documents based on extreme value theory. Compared with the method of sorting documents using LLM, EVT-Rank has lower reordering delay, and can be deployed on the CPU side to avoid potential memory overflow on GPU.

## Environment
We developed EVT-Rank using Python, utilizing version 3.10.12 for our implementation.

### Hardware and Software
The deployment of EVT-Rank was performed on two servers with different configurations, and the parameters are presented in the table below.

| Configuration | Server 1 Parameters | Server 2 Parameters |
| :--------: | :------: | :------: |
| CPU   | Intel Xeon Bronze 31xx Series | 16-Core AMD EPYC 9354 |
| GPU | NVIDIA GeForce RTX 3060 (12GB VRAM) | NVIDIA GeForce RTX 4090 (25.2GB VRAM) |
| Memory | 64GB DDR4 | 60.1 GB |
| Primary Storage	 | 1TB NVMe SSD | 700GB NVMe SSD |
| Secondary Storage	 | 2TB HDD | None |
| Operating System	 | Linux | Linux |
| Acceleration Environment	 | CUDA 12.8 | CUDA 13.0 |
| Development Platform	 | Visual Studio Code 1.103.2 | Visual Studio Code 1.103.2 |

### Library
The Python libraries and version we used are listed below.

| Lib | Version |
| :--------: | :------: |
| PyTorch   | 2.10.0 |
| Transformers | 5.6.2 |
| Pandas	 | 2.3.3 |
| SciPy	 | 1.15.3 |
| NumPy	 | 2.2.6 |

## Datasets

### QA Dataset
We evaluated the performance of EVT-Rank on both medical and generaldomain question-answering (QA) datasets. 
For the medical domain, five datasets from the MIRAGE benchmark were selected, totaling 7,663 questions. These include three medical examination datasets (MMLU-Med, MedQAUS, and MedMCQA) and two biomedical QA datasets (PubMedQA* and BioASQ-Y/N). These five datasets are publicly accessible at https://github.com/gzxiong/MIRAGE/tree/main. 
For the general domain, we utilized the MMLU dataset, which covers 57 subjects across various difficulty levels. The MMLU dataset can be obtained from https://modelscope.cn/datasets/cais/mmlu/files.

### Corpus for Retriever
For medical domain questions, MedCorp is utilized as the external document library of the RAG system. Its original text data originates from four distinct sources: PubMed (a biomedical abstract corpus), StatPearls (a clinical decision support text suite), Textbooks (a collection of domain-specific medical textbooks), and Wikipedia (a general knowledge corpus). MedCorp is accessible at https://github.com/gzxiong/MedRAG. 
For general-domain questions in the MMLU dataset, we restrict the external document library for retrieval exclusively to the Wikipedia component within MedCorp.

## Retriever
We employ Contriever as the retrieval module of our RAG system, which maps text into 768-dimensional dense embeddings and retrieves the most relevant documents from the external library by computing vector similarity. The Contriever model is publicly accessible at https://modelscope.cn/models/facebook/contriever/files.

## Downstream LLM 
To simulate resource-constrained application scenarios, Qwen3-0.6B and Qwen3-4B were selected as the downstream LLM. These models can be downloaded from the following websites: https://modelscope.cn/models/Qwen/Qwen3-0.6B/files and https://modelscope.cn/models/Qwen/Qwen3-4B/files.

## Implementation

### Generate Passage Embeddings (`generate_passage_embeddings.py`)
Run this script to preprocess the raw texts in MedCorp and generate 768-dimensional semantic embedding vectors for each text chunk. This preprocessing step enables the dense retriever to perform efficient similarity searches between user queries and the document corpus.

### Compute Corpus Mean Vector (`mean_vector.py`)
This script calculates the mean vector across all semantic embeddings in the external corpus to establish its semantic center. In this framework, a lower similarity between a document's embedding and the mean vector indicates that the document contains more outlier-like or rare knowledge.

### Long-tail Mapping (`longtail_mapping.py`)
Use this script to map the semantic vectors of documents into numerical scarcity scores. A higher score quantifies that the corresponding text contains scarcer, long-tail knowledge within the context of the corpus.

### EVT Applicability Check (`EVT_check.py`)
This module evaluates whether the mapped scarcity score sequence satisfies the theoretical assumptions of Extreme Value Theory (EVT). It automatically performs:
- **Stationarity Testing**
- **Block Maxima Processing**
- **Extremal Independence Testing**
> **Note on Acceptance Criteria:** The sequence is considered suitable for EVT modeling when the KPSS statistic is below `0.74` and the Berman statistic is below `0.2`.
