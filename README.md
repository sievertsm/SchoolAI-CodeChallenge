# SchoolAI Code Challenge

## Document Embedding and RAG Implementation

## Objective

Develop a simple question-answering system using a Retrieval-Augmented Generation (RAG) framework. The system should embed documents from a public dataset, retrieve relevant information based on a user query, and generate an appropriate response.

## Approach

The approach to this project included the following steps:

### 1. Data Preparation

This project utilized the [SciQ](https://huggingface.co/datasets/sciq) dataset from Hugging Face, which contains crowdsourced science exam questions. The `support` feature, which contains supporting evidence for the correct answer was used for the documents in the RAG framework.

A subset of 1000 entries of this dataset was used in the RAG framework. The `support` feature was not present in every entry in the dataset, so the entries which did not have the `support` feature were removed prior to sampling. Also, the `support` feature was cleaned by attempting to remove all links. 

![image]("data/word-count-distribution.png")

### 2. Document Embedding

The sample of cleaned support documents were embedded using the [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model. This model encodes text into a embedding with 384 features. These embeddings were then stored in a vector database using [`faiss`](https://faiss.ai/index.html#faiss), which efficiently stores the embeddings for search and retrieval. 

### 3. RAG Framework Implementation

The retrieval mechanism used to find relevent documents based on a query included embedding the query using the [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model, and then using that embedding to search the [`faiss`](https://faiss.ai/index.html#faiss) database for the top-k documents based on cosine similarity.

The selection of retrieved documents was refined using [`kneed`](https://kneed.readthedocs.io/en/stable/api.html#kneelocator) (kneedle) dynamic thresholding and by re-ranking the documents using cosine similarity based on the TF-IDF features [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#tfidfvectorizer).

The query combined with the retrieved and ranked documents were used to construct a prompt for a large language model to answer a question based on the provided information. The model used was [`Llama 70b`](https://huggingface.co/meta-llama/Meta-Llama-3-70B) using Groq.

### 4. Query Interface

A Gradio interface was used to interact with the RAG framework.

## Instructions for Use

### Environment
First an environment must be set up to run the code, which can be completed by following similar commands as these:

Create a virtual environment
```
python -m venv .env
```

Activate the environment
```
source .env/bin/activate
```

Install the required packages
```
pip install -r requirements.txt
```

Install the SpaCy English resources
```
python -m spacy download en_core_web_sm
```

### RAG System
The RAG system can be run in the configured environment with the command
```
python app.py
```

This app launches a Gradio interface to interact with the RAG system. The app uses the class `QuestionAnswerRAG` which is in the question_answer_rag.py file. 


## Challenges

I encountered some challenges while working on this project

- One of the challenges was removing links from the text in the dataset. Many of the links were broken up with white-space which made it so I wasn't able to remove them using regular expressions.
- Another challenge was getting Llama 70b working in my code using HuggingChat. I wasn't able to find a way to get this working without a paid version of HuggingFace, so I found a way to leverage this model using Groq instead.


## Potential Improvements

This was an interesting project that resulted in a simple RAG framework. There are many ways this framework could potentially be optimized or improved. These are some potential area for improvement. 

- **Encoding Model Selection** In this project I selected the [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model because it was the most liked model on HuggingFace. It would be interesting to try different encoding models in this framework.
- **Dynamic Thresholding** Experiemnting with how different dynamic thresholding techniques perform could improve performance. 
- **Dimensionality Reduction** It would be interesting to see if performance improves or deteriorates with dimensionality reduction. 
- **LLM Model Selection** Comparing the output from different models would be interesting. For instance, it would be interesting to compare differences in performance on this task between the Llama 8b model and the Llama 70b model.  
- **LLM Hyperparameter Tuning** Experiementing with the LLM parameters, such as temperature, to try and optimize the output could improve the output.
- **Prompt Engineering** By changing the content of the prompt the RAG framework passes to the LLM could improve performance.
 







