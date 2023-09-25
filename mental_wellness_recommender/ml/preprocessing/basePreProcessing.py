from abc import ABC, abstractmethod


class BasePreProcessing(ABC):

    @abstractmethod
    def preProcess(self, df):
        pass
