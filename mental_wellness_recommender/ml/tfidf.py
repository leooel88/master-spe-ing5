import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
from .base_model import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))


class TfidfModel(BaseModel):

    def __init__(self, resources_df=None):
        self.resources = resources_df

    @property
    def name(self):
        return "SVC"

    @property
    def recommendation_number(self):
        return 5

    def train(self, pkl_path, resources_df=None):
        if resources_df is not None:
            self.resources = resources_df

        X = self.resources['description']

        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        X_vect = vectorizer.fit_transform(X)

        with open(pkl_path, 'wb') as f:
            pickle.dump((X_vect, vectorizer, self.resources), f)

        print('TF-IDF matrix created and saved as tfidf_matrix.pkl')

    def recommend(self, pkl_path, user_input):
        with open(pkl_path, 'rb') as f:
            X_vect, vectorizer, resources = pickle.load(f)

        user_input_vect = vectorizer.transform([user_input])
        cosine_similarities = cosine_similarity(user_input_vect, X_vect)
        top_k_indices = cosine_similarities.argsort(
        ).flatten()[-self.recommendation_number:]

        recommended_resources = resources.iloc[top_k_indices].sort_values(
            by='title', ascending=True
        )

        return recommended_resources
