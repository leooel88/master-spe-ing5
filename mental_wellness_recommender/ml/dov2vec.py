# Doc2VecModel.py

import os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .base_model import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))


class Doc2VecModel(BaseModel):

    def __init__(self, resources_df=None):
        self.resources = resources_df

    @property
    def name(self):
        return "Doc2Vec"

    @property
    def recommendation_number(self):
        return 5

    def train(self, pkl_path, resources_df=None):
        if resources_df is not None:
            self.resources = resources_df

        tagged_data = [TaggedDocument(words=desc.split(), tags=[str(i)])
                       for i, desc in enumerate(self.resources['description'])]

        model = Doc2Vec(vector_size=100, window=5,
                        min_count=1, workers=4, epochs=100)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count,
                    epochs=model.epochs)

        model.save(pkl_path)
        print('Doc2Vec model trained and saved.')

    def recommend(self, pkl_path, user_input):
        model = Doc2Vec.load(pkl_path)

        # Transform user input
        user_vector = model.infer_vector(user_input.split())

        # Calculate cosine similarity
        cosine_sims = np.array([cosine_similarity([user_vector], [model.dv[i]])[0][0]
                                for i in range(len(self.resources))])

        # Get top-k recommendations
        top_k_indices = cosine_sims.argsort(
        )[-self.recommendation_number:][::-1]
        recommended_resources = self.resources.iloc[top_k_indices].sort_values(
            by='title', ascending=True)

        return recommended_resources
