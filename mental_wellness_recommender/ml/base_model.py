from abc import ABC, abstractmethod


class BaseModel(ABC):

    @property
    @abstractmethod
    def name(self, resources_df=None):
        pass

    @abstractmethod
    def train(self, pkl_path, resources_df=None):
        pass

    @abstractmethod
    def recommend(self, pkl_path, user_input):
        pass
