import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from .base_model import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))


class KnnModel(BaseModel):

    @property
    def name(self, resources_df=None):
        return "KNN"

    def train(self, pkl_path, resources_df=None):
        if resources_df is not None:
            self.resources = resources_df

        self.resources = self.resources.dropna(subset=['description'])
        vectorizer = TfidfVectorizer(stop_words='english')
        resources_tfidf_matrix = vectorizer.fit_transform(
            self.resources['description'])

        X = resources_tfidf_matrix
        y = self.resources['resource_id']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=1))

        model_dict = {
            'knn': knn,
            'vectorizer': vectorizer
        }

        with open(pkl_path, 'wb') as f:
            pickle.dump(model_dict, f)

    def recommend(self, pkl_path, user_input):
        with open(pkl_path, 'rb') as f:
            model_dict = pickle.load(f)

        # Load the saved vectorizer and KNN model
        knn = model_dict['knn']
        vectorizer = model_dict['vectorizer']

        input_tfidf = vectorizer.transform([user_input])
        resource_id = knn.predict(input_tfidf)[0]

        return self.resources.loc[self.resources['resource_id'] == resource_id]
