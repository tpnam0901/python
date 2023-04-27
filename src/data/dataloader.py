import random
from abc import ABC, abstractmethod
from typing import List

from keras.utils import Sequence


class KerasSequence(ABC, Sequence):
    def __init__(
        self,
        data_list: List,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        """Keras Sequence class for data loading

        Args:
            data (List): List of data
            batch_size (int, optional): Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        """
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Return the number of batches of this dataset."""
        return len(self.data_list) // self.batch_size

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.data_list))
        batch = self.data_list[low:high]
        return self.preprocess(batch)

    @abstractmethod
    def preprocess(self, batch):
        """Preprocess the batch and return the processed batch"""
        return batch

    def on_epoch_end(self) -> None:
        """
        Updates indexes after each epoch
        """
        if self.shuffle:
            random.shuffle(self.data_list)
