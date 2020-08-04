import os.path
import numpy as np
import h5py

from utils import log


class SortOfCLEVR(object):
    def __init__(self, path, split='train'):
        filename = 'data.hy'
        file = os.path.join(path, filename)
        log.info("Reading %s ...", file)
        try:
            data = h5py.File(file, 'r')
        except:
            raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
        log.info("Reading Done: %s", file)

        self.split = split
        self._data = data[split]

    def query_dataset(self):
        # preprocessing and data augmentation
        img = self._data['image'][()]
        q = self._data['question'][()]
        a = self._data['answer'][()]
        return img, q, a

    def __len__(self):
        return len(self._data['image'])

    def __repr__(self):
        return 'Sort-of-CLEVR (%s, %d examples)' % (self.split, len(self))

