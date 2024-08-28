from question_answer_rag import QuestionAnswerRAG
import gradio as gr

from datasets import load_dataset
import numpy as np
import pandas as pd
import re

def clean_text(text):
    """
    """
    # remove links
    text = re.sub(r"http\S+", "", text) # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python

    return text

def get_data():

    # load the "sciq" dataset from Hugging Face
    ds = load_dataset("allenai/sciq") # load the dataset
    ds_train = ds["train"] # subset the training set
    
    # create dataframe for data manipulation
    df = pd.DataFrame(ds_train)

    # apply the clean text function
    df["support-clean"] = df["support"].apply(clean_text)
    
    # investigate word count (split on whitespace)
    df["word_count"] = df["support-clean"].apply(lambda x: len(x.split()))

    # filter out rows with too few words
    min_word_count=5
    df_filt = df[df["word_count"]>=min_word_count].copy().reset_index() # original index will be stored as a feature

    # select a subset of 1000 documents from the dataset

    # set number of samples to be selected
    n_samples=1000
    
    # set random seed for reproducibility
    np.random.seed(147)
    # get a set of random indicies to subset the data
    idx_sample = np.random.choice(len(df_filt), size=n_samples, replace=False)
    
    # subset the dataset
    df_train_sample = df_filt.iloc[idx_sample, :].reset_index(drop=True)
    
    print(f"Total samples in subset: {len(df_train_sample)}")

    return df_train_sample
    

def main():

    df_train_sample = get_data()

    # get the support-clean column of the data to use as text embeddings
    support_feat = df_train_sample["support-clean"].values
    
    # instantiate the QuestionAnswerRAG class
    rag = QuestionAnswerRAG()
    # create vector database
    rag.create_vector_database(support_feat)
    
    def chat_question_answer_rag(question, history):
        return rag.question_answer_rag(question)

    gr.ChatInterface(
        chat_question_answer_rag,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask me a science related question", container=False, scale=7),
        title="SciQ Chat",
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
        theme="base",
    ).launch(share=False)

if __name__ == "__main__":
    main()