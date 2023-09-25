# PreProcessing2.py

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser
from .basePreProcessing import BasePreProcessing

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))


class PreProcessing2(BasePreProcessing):

    def __init__(self):
        self.bigram = None

    def lemmatize_text(self, text):
        return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

    def clean_text(self, text):
        return ' '.join([char.lower() for char in word_tokenize(text) if char.isalpha()])

    def remove_stopwords(self, text):
        return ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    def apply_bigrams(self, sentences):
        phrases = Phrases(sentences, min_count=5, threshold=7)
        self.bigram = Phraser(phrases)
        return [self.bigram[sent] for sent in sentences]

    def preProcess(self, resources_df):
        resources_df['description'] = resources_df['description'].apply(
            self.clean_text)
        resources_df['description'] = resources_df['description'].apply(
            self.remove_stopwords)
        resources_df['description'] = resources_df['description'].apply(
            self.lemmatize_text)

        descriptions = [desc.split() for desc in resources_df['description']]
        bigram_descriptions = self.apply_bigrams(descriptions)
        resources_df['description'] = [
            ' '.join(desc) for desc in bigram_descriptions]

        return resources_df
