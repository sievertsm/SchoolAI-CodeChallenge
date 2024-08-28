from question_answer_rag import QuestionAnswerRAG
from question_answer_rag import get_sampled_data
import gradio as gr

import numpy as np
import pandas as pd


def main():

    # df_train_sample = get_sampled_data()
    df_train_sample = pd.read_csv("data/sciq-train-sampled.csv")

    # get the support-clean column of the data to use as text embeddings
    support_feat = df_train_sample["support-clean"].values
    
    # instantiate the QuestionAnswerRAG class
    rag = QuestionAnswerRAG()
    # create vector database
    rag.create_vector_database(support_feat)
    
    def chat_question_answer_rag(question, history):
        """
        Formats RAG output for the chat interface
        """

        answer, documents = rag.question_answer_rag(question)
        
        text_out = 35 * "-" + " Top Documents " + 35 * "-" + "\n\n"
        for i, d in enumerate(documents): 
            text_out += f"Rank {i+1} Document: {d}\n\n"
        text_out += "\n\n" + 39 * "-" + " Answer " + 39 * "-" + "\n\n"
        text_out += answer
    
        return text_out

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