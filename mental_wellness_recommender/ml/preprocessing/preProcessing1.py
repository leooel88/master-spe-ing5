import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from .basePreProcessing import BasePreProcessing

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))


class PreProcessing1(BasePreProcessing):

    def lemmatize_text(self, text):
        words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def clean_text(self, text):
        # Supprimer les caractères non alphabétiques et mettre le texte en minuscules
        text = ''.join(
            [char for char in text if char.isalpha() or char.isspace()])
        text = text.lower()
        return text

    def remove_stopwords(self, text):
        words = nltk.word_tokenize(text)
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)

    def preProcess(self, resources_df):
        resources_df['description'] = resources_df['description'].apply(
            self.clean_text)
        resources_df['description'] = resources_df['description'].apply(
            self.remove_stopwords)
        resources_df['description'] = resources_df['description'].apply(
            self.lemmatize_text)
        return resources_df
