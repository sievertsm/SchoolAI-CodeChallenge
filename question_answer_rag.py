from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import pandas as pd
import faiss
import re
from kneed import KneeLocator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

class QuestionAnswerRAG:
    
    def __init__(
        self, 
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
        top_k = 10
    ):
    
        # instantiate embedding model
        self.model_embed = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.source_text = None
        self.index = None
        self.nfeatures_embedding = None

    def embed_text(self, text):
        """
        Text is embedded using the selected SentenceTransformer model
        """
        return self.model_embed.encode(text)


    def create_vector_database(self, text_list):
        """
        A FAISS vector database is created using the input 'text_list.' The vector database uses
        normalized embeddings and inner products to return cosine similarity during search.
        """

        # convert text_list to np.array for ease of indexing
        self.source_text = np.array(text_list)

        # get the text embeddings 
        embeddings_faiss = self.embed_text(text_list)

        # store the number of features in the embedding 
        self.nfeatures_embedding = embeddings_faiss.shape[1]

        # store embeddings in the vector database
        faiss.normalize_L2(embeddings_faiss)  # normalize embeddings for cosine similarity
        index = faiss.IndexFlatIP(embeddings_faiss.shape[1]) # inner product for cosine similarity
        index.add(embeddings_faiss)

        # store the databse in the class
        self.index = index


    def query_vector_database(self, text):
        """
        Returns the top-k entries from the vector database that match the input text
        """

        # embedd the text
        vector = self.embed_text(text)

        # reshape the text to have the dimensions (N, num_features)
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)

        # search the vector database for similar entries
        faiss.normalize_L2(vector)  # Normalize embeddings for cosine similarity
        similarities, indices = self.index.search(vector, self.top_k) 

        return similarities[0], indices[0]

    
    def dynamic_threshold(self, similarities, indicies):
        """
        Dynamically thresholds the retrieved documents using the 'kneedle' method on the cosine similarities.
        """
        # get a range of x-values that corresponds to the length of the cosine similarites
        x = np.arange(len(similarities))

        # locate the knee of the similarites
        kneedle = KneeLocator(x, similarities, S=1.0, curve="convex", direction="decreasing")
        knee = kneedle.knee

        # if no knee is identified return un-thresholded data
        if not knee:
            return similarities, indicies

        # threshold the data
        knee_th = knee + 1 # add 1 for indexing
        similarities_th, indicies_th = similarities.copy(), indicies.copy() # copy to ensure original data is unchanged
        similarities_th, indicies_th = similarities_th[:knee_th], indicies_th[:knee_th] # threshold
    
        return similarities_th, indicies_th

    def reranking_tfidf(self, query_text, source_material):
        """
        Re-ranks the retrieved documents using term frequency inverse document frequency (tfidf)
        """

        # concatenates the query with the source material
        t = np.concatenate([np.array([query_text]), source_material])

        # instantiates the TfidfVectorizer
        tfidfvectorizer = TfidfVectorizer()

        # fit and transform the vectorizer
        X = tfidfvectorizer.fit_transform(t)
        # get similarity values comparing the query to the support documents
        X_similarity = cosine_similarity(X)[0, 1:]

        # get indecies that sort the similarity measures (high to low)
        idx_sort = np.argsort(X_similarity)[::-1]

        # re-order the source material
        source_material_ranked = source_material[idx_sort]
    
        return source_material_ranked


    def construct_rag_prompt(self, query_text):
        """
        Constructs the prompt passed to the LLM by combining other functions within the class.
        """
        # return similar documents to the query
        similarities, indicies = self.query_vector_database(query_text)
        
        # threshold search results
        similarities_th, indicies_th = self.dynamic_threshold(similarities, indicies)

        if len(indicies_th) > 0:
            # retrieve relevent source text
            source_material = self.source_text[indicies_th]
            # re-rank documents
            source_material = self.reranking_tfidf(query_text, source_material)
        else:
            # if there are no indicies there are no documents to use
            source_material = ['']

        # construct the query
        # prompt_text = f"{query_text}\n\nDocuments:\n"
        # add the prompt text as the question portion of the query
        prompt_text = f"{query_text}\n\n"
        
        # add a prefix so the model understands these are documents to answer the question
        document_prefix = "These documents have been retrieved and ranked by relevence:"
        prompt_text += f"{document_prefix}\n\n"
        
        # iterate over the items of source material and add them to the prompt
        for i, s in enumerate(source_material):
            # prompt_text += f"{s}\n\n"
            prompt_text += f"Rank {i+1} Document: {s}\n---\n"

        # remove the final new line
        prompt_text = prompt_text[:-2]
    
        return prompt_text


    def question_answer_rag(self, question):
        """
        Retrieves an answer to a question using an LLM and RAG
        """
        
        # gets the RAG prompt based on the question
        rag_prompt = self.construct_rag_prompt(question)

        # sets up the Groq client
        groq_api = "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"
        client = Groq(api_key=groq_api)

        # gets an answer to the question using a llama3 model
        completion = client.chat.completions.create(
            # model="llama3-8b-8192",
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": rag_prompt
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # extract the answer to the question
        msg = completion.choices[0].message
        answer = msg.content

        return answer