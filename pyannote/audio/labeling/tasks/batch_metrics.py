from typing import List, Dict, Tuple
import yaml
import ipdb
import bisect
import numpy as np
import scipy.signal
#from pyannote.audio.features import FeatureExtraction
from pyannote.audio.features.utils import get_audio_duration
from pyannote.core import SlidingWindow, Segment
from pyannote.core.utils.helper import get_class_by_name
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.database import get_protocol, FileFinder
from pyannote.database.util import LabelMapper
from pyannote.database import get_unique_identifier


class Batch_Metrics():
    """log some metrics on the batch generation"""
    def __init__(self, all_labels: list, batch_size:int, batch_log:str, frame_info, data_, duration: int):

        self.all_labels = all_labels
        self.batch_log = batch_log
        self.duration = duration
        self.window = frame_info
        self.batch_size = batch_size

        # Initialize log
        if self.batch_log:
            with open(self.batch_log, 'w') as fout:
                fout.write(u'sample number,batch number,coverage,{},{}\n'.format(','.join([label + ' coverage' for label in self.all_labels]),
                                                                               ','.join([label + ' batch balance' for label in self.all_labels]),
                                                                               ','.join([label + ' whole balance' for label in self.all_labels])))
        # get 1hot per label
        file_set = [data_[uri] for uri in data_]
        labels_one_hot = [file['y'].data.swapaxes(0,1).astype(np.bool) for file in file_set]
        self.files_duration = [one_hot.shape[1] for one_hot in labels_one_hot]
        self.allFiles_one_hot = np.concatenate(labels_one_hot, axis=1)
        self.corpus_length = self.allFiles_one_hot.shape[1]
        self.label_frequencies = np.sum(self.allFiles_one_hot, axis = 1)

        # keep beginning/end of each file
        indexes_ranges = [0]
        for i, file in enumerate(file_set):
            indexes_ranges.append(indexes_ranges[i] + self.files_duration[i])
        self.indexes_ranges = np.array(indexes_ranges)
        self.uris2idx = {file['current_file']['uri']: self.indexes_ranges[i] for i, file in enumerate(file_set)}

        # init metrics
        self.whole_coverage_one_hot = np.zeros((1, self.allFiles_one_hot.shape[1]))
        self.label_coverage_one_hot = np.zeros((len(self.all_labels), self.allFiles_one_hot.shape[1]))
        self.batch_num = -1
        self.batch_idx = 0 # index of the batch change in self.batch_one_hot
        self.coverage = 0
        self.whole_coverage = np.zeros((1,1))
        self.label_coverage_one_hot = np.zeros(self.allFiles_one_hot.shape)
        self.label_coverage = np.zeros((len(all_labels), ))
        self.batch_one_hot = np.zeros((len(all_labels), 1))
        
        self.balance = dict()
        self.run_balance = dict() # class balance over the whole run, not just per batch
        for label in self.all_labels:
            self.balance[label] = []
            self.run_balance[label] = []
        
    def _abs2rel_timestamps(self, abs_onset: int, abs_offset: int, uri: str):
        """ Convert timestamps in seconds relative to the file to 
            timestamps in samples relative to the corpus
        """

        # get wav name from uri
        uri = uri.split('/')[1]

        # get timestamps in samples
        abs_onset_ = abs_onset / self.window.duration
        abs_offset_ = abs_offset / self.window.duration

        # get timestamps relative to the whole concatenated corpus
        ## look out, sometimes  occurs in float, int just does "floor", not round
        rel_onset = int(np.ceil(abs_onset_ + self.uris2idx[uri]))
        rel_offset = int(np.ceil(abs_offset_ + self.uris2idx[uri]))
        return rel_onset, rel_offset

    def compute_coverage(self, abs_onset: int, abs_offset: int, uri: str):
        """ Given the timestamps of a segment, add it too the coverage
            Input onset/offset are for the file, rel_onset/offset (for relative)
            are in the whole one hot of the whole corpus"""

        # convert timestamps into samples
        rel_onset, rel_offset = self._abs2rel_timestamps(abs_onset, abs_offset, uri)

        ## TODO DO THAT FOR EACH LABEL
        self.whole_coverage_one_hot[0,rel_onset:rel_offset] += 1
        cov = np.sum(np.minimum(self.whole_coverage_one_hot, np.ones(self.whole_coverage_one_hot.shape))) / self.corpus_length
        self.whole_coverage = np.concatenate([self.whole_coverage, np.array([[cov]])], axis=1)

        # increment the batch number at each batch
        if self.whole_coverage.shape[1] % self.batch_size == 0:
            self.batch_num += 1
            self.batch_idx = self.batch_one_hot.shape[1]
        return cov

    def compute_balance(self, abs_onset: int, abs_offset: int, uri: float):
        """For each label compute it's frequency in the batch"""

        # get timestamps
        rel_onset, rel_offset = self._abs2rel_timestamps(abs_onset, abs_offset, uri)

        # get one hot of segment
        sample_one_hot = self.allFiles_one_hot[:, rel_onset: rel_offset]
        self.label_coverage_one_hot[:, rel_onset: rel_offset] = sample_one_hot
        self.label_coverage = np.sum(self.label_coverage_one_hot, axis=1) / self.label_frequencies

        # append one hot of sample to log of all samples
        self.batch_one_hot = np.concatenate([self.batch_one_hot, sample_one_hot], axis=1)

        # compute balance on batch level
        sample_size = self.window.durationToSamples(self.duration)
        nb_samples = self.batch_one_hot[:, self.batch_idx :].shape[1]
        for i, label in enumerate(self.all_labels):
            # batch level
            self.balance[label].append(float(np.sum(self.batch_one_hot[i, self.batch_idx:])) / nb_samples)
            #whole run level
            self.run_balance[label].append(float(np.sum(self.batch_one_hot[i, :])) / self.batch_one_hot.shape[1])
        return self.balance        

    def dump_stats(self, output: str, with_balance: bool=True):
        """Call to write csv output with stats.
           Put balance to False to log only coverage"""

        with open(output, 'a') as fout:
            for i in range(self.batch_num*self.batch_size, (self.batch_num + 1) * self.batch_size):
                # don't write balance when random sampling yet ## TODO Implement label retrieval for random sampling
                if with_balance: 
                    fout.write(u'{},{},{},{},{},{}\n'.format(str(i), str(i // self.batch_size), str(self.whole_coverage[0,i]), 
                                                       ','.join([str(self.label_coverage[i]) for i, _ in enumerate(self.all_labels)]),
                                                       ','.join([str(self.balance[label][i]) for label in self.balance]),
                                                       ','.join([str(self.run_balance[label][i]) for label in self.run_balance])))
                else:
                    fout.write(u'{},{},{},{}\n'.format(str(i), str(i // self.batch_size), str(self.whole_coverage[0,i]),
                                                       ','.join([str(self.label_coverage[i]) for i, _ in enumerate(self.all_labels)])))

