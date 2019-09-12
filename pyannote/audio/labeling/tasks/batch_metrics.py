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
    def __init__(self, protocol, subset: str, all_labels: list, batch_size:int, batch_log:str):

        self.all_labels = all_labels
        self.batch_log = batch_log

        if self.batch_log:
            with open(self.batch_log, 'w') as fout:
                fout.write(u'sample number,batch number,coverage,{},{}\n'.format(','.join([label + '_cov' for label in self.all_labels]),
                        
                                                                               ','.join([label + '_bal' for label in self.all_labels])))

        ## TODO DEFINE THIS PRETTYLY 
        self.window = SlidingWindow(duration=0.01, step=0.01, start=0.0)

        self.batch_size = batch_size

        # get 1hot per label
        file_set = list(getattr(protocol, subset)())
        
        labels_one_hot = [one_hot_encoding(file['annotation'],
                                            file['annotated'],
                                            self.window,
                                            labels=self.all_labels)[0].data.swapaxes(0,1) for file in file_set]
        self.files_duration = [one_hot.shape[1] for one_hot in labels_one_hot]
        self.allFiles_one_hot = np.concatenate(labels_one_hot, axis=1)
        self.whole_coverage_one_hot = np.zeros((1, self.allFiles_one_hot.shape[1]))
        self.label_coverage_one_hot = np.zeros((len(self.all_labels), self.allFiles_one_hot.shape[1]))


        # keep beginning/end of each file
        indexes_ranges = [0]
        for i, file in enumerate(file_set):
            indexes_ranges.append(indexes_ranges[i] + self.files_duration[i])
        self.indexes_ranges = np.array(indexes_ranges)
        self.uris2idx = {file['uri']: self.indexes_ranges[i] for i, file in enumerate(file_set)}
        #self.label2idx = {label: i for i, label in self.all_labels}

        # init metrics
        self.batch_num = -1
        self.coverage = 0
        self.whole_coverage = np.zeros((1,1))
        self.label_coverage_one_hot = np.zeros(self.allFiles_one_hot.shape)
        self.label_coverage = np.zeros((len(all_labels), ))
        self.batch_one_hot = np.zeros((len(all_labels), 1))
        
        self.balance = dict()
        for label in self.all_labels:
            self.balance[label] = []
        

    def _get_sample_one_hot(self, abs_onset: int, abs_offset: int, uri: str):
        """Get the one hot for each label for the picked sample"""
        # get position of file in corpus_one_hot
        ## TODO PRETTIER THAN THAT ! 
        uri = uri.split('/')[1]

        rel_onset = abs_onset + self.uris2idx[uri]
        rel_offset = abs_offset + self.uris2idx[uri]
        return rel_onset, rel_offset
        #return self.allFiles_one_hot[:, rel_onset: rel_offset]

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
        # put timestamps into samples
        rel_onset, rel_offset = self._abs2rel_timestamps(abs_onset, abs_offset, uri)

        ## TODO DO THAT FOR EACH LABEL
        self.whole_coverage_one_hot[0,rel_onset:rel_offset] += 1
        cov = np.sum(np.minimum(self.whole_coverage_one_hot, np.ones(self.whole_coverage_one_hot.shape))) / self.whole_coverage_one_hot.shape[1]
        self.whole_coverage = np.concatenate([self.whole_coverage, np.array([[cov]])], axis=1)
        # increment the batch number at each batch
        if self.whole_coverage.shape[1] % self.batch_size == 0:
            self.batch_num += 1
        return cov

    def compute_balance(self, label: str, abs_onset: int, abs_offset: int, uri: float):
        """For each label compute it's frequency in the batch"""
        rel_onset, rel_offset = self._abs2rel_timestamps(abs_onset, abs_offset, uri)
        # get one hot of segment
        sample_one_hot = self.allFiles_one_hot[:, rel_onset: rel_offset]
        self.label_coverage_one_hot[:, rel_onset: rel_offset] = sample_one_hot ## TODO ADD TO GET DENSITY ? MEH NOT NOW ... 
        self.label_coverage = np.sum(self.label_coverage_one_hot, axis=1) / np.sum(self.allFiles_one_hot, axis = 1)

        self.batch_one_hot = np.concatenate([self.batch_one_hot, sample_one_hot], axis=1)

        # compute balance
        nb_samples = self.batch_one_hot.shape[1]
        if label:
            for i, label in enumerate(self.all_labels):
                self.balance[label].append(float(np.sum(self.batch_one_hot[i, :])) / nb_samples)
        return self.balance        

    def dump_stats(self, output: str, with_balance: bool=True):
        """Call to write csv output with stats.
           Put balance to False to log only coverage"""
        ### TODO SILLY OVERWRITES THE FILE EACH TIME 
        #print('batch number {}'.format(self.batch_num))
        with open(output, 'a') as fout:
            #fout.write(u'sample number,batch number,coverage,{}\n'.format(','.join([label for label in self.balance])))
            for i in range(self.batch_num*self.batch_size, (self.batch_num + 1) * self.batch_size):
                # batch index is integer division of sample number and batch size
                # get label coverage
                if with_balance: 
                    fout.write(u'{},{},{},{},{}\n'.format(str(i), str(i // self.batch_size), str(self.whole_coverage[0,i]), 
                                                       ','.join([str(self.label_coverage[i]) for i, _ in enumerate(self.all_labels)]),
                                                       ','.join([str(self.balance[label][i]) for label in self.balance])))
                else:
                    fout.write(u'{},{},{},{}\n'.format(str(i), str(i // self.batch_size), str(self.whole_coverage[0,i]),
                                                       ','.join([str(self.label_coverage[i]) for i, _ in enumerate(self.all_labels)])))


        ## write CSV with coverage density
        #lower = self.batch_num * self.batch_size
        #upper = (self.batch_num + 1) * self.batch_size
        #with open(output + '_cov_{}.csv'.format(self.batch_num), 'w') as fout:
        #    density = list(scipy.signal.decimate(self.whole_coverage_one_hot[0,:],10))
        #    fout.write(u'{}\n'.format(','.join([str(d) for d in density])))

    
    def generation_time():

        pass
