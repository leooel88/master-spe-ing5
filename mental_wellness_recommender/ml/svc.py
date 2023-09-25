import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import os
from .base_model import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))


class SvcModel(BaseModel):

    @property
    def name(self):
        return "SVC"

    @property
    def recommendation_number(self):
        return 5

    def train(self, pkl_path, resources_df=None):
        if resources_df is not None:
            self.resources = resources_df

        self.resources = self.resources.dropna(subset=['description'])

        X = self.resources['description']
        y = self.resources['category']
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        X_train_vect = vectorizer.fit_transform(X_train)

        svm_model = LinearSVC()
        svm_model.fit(X_train_vect, y_train)

        with open(pkl_path, 'wb') as f:
            pickle.dump((svm_model, vectorizer), f)

        print('SVM model trained and saved as svc_model.pkl')

    def recommend(self, pkl_path, user_input):
        with open(pkl_path, 'rb') as f:
            svm_model, vectorizer = pickle.load(f)

        user_input_vect = vectorizer.transform([user_input])
        predicted_category = svm_model.predict(user_input_vect)[0]

        recommended_resources = self.resources[self.resources['category'] == predicted_category].sort_values(
            by='title', ascending=True
        )

        return recommended_resources[-self.recommendation_number:]
