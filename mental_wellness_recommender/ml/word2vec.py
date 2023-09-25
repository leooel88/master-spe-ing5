import os
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .base_model import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))


class Word2VecModel(BaseModel):

    def __init__(self, resources_df=None):
        self.resources = resources_df

    @property
    def name(self):
        return "Word2Vec"

    @property
    def recommendation_number(self):
        return 5

    def train(self, pkl_path, resources_df=None):
        if resources_df is not None:
            self.resources = resources_df

        # Pretraitement des descriptions pour les transformer en listes de mots
        sentences = [desc.split() for desc in self.resources['description']]

        # Entraînement du modèle Word2Vec
        model = Word2Vec(sentences, vector_size=100,
                         window=5, min_count=1, workers=4)
        model.save(pkl_path)
        print('Word2Vec model trained and saved.')

    def recommend(self, pkl_path, user_input):
        model = Word2Vec.load(pkl_path)

        user_input_words = [
            word for word in user_input.split() if word in model.wv.index_to_key]
        if not user_input_words:
            return []

        user_input_vect = np.array([model.wv[word]
                                   for word in user_input_words])
        avg_vector = user_input_vect.mean(axis=0)

        descriptions = self.resources['description'].tolist()
        desc_vectors = []

        for desc in descriptions:
            desc_words = [word for word in desc.split(
            ) if word in model.wv.index_to_key]
            if not desc_words:
                desc_vectors.append(np.zeros(model.vector_size))
                continue
            desc_vect = np.array([model.wv[word] for word in desc_words])
            avg_desc_vector = desc_vect.mean(axis=0)
            desc_vectors.append(avg_desc_vector)

        desc_vectors_matrix = np.vstack(desc_vectors)
        cosine_sims = cosine_similarity([avg_vector], desc_vectors_matrix)[0]

        top_k_indices = cosine_sims.argsort(
        )[-self.recommendation_number:][::-1]

        recommended_resources = self.resources.iloc[top_k_indices].sort_values(
            by='title', ascending=True)

        return recommended_resources
