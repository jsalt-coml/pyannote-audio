#!/usr/bin/env python
# encoding:  utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions: 

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Marvin Lavechin - marvinlavechin@gmail.com

"""
Multi-label classifier BabyTrain

Usage: 
  pyannote-multilabel train [options] <experiment_dir> <database.task.protocol>
  pyannote-multilabel validate [options] [--every=<epoch> --chronological --precision=<precision> --use_der] <label> <train_dir> <database.task.protocol>
  pyannote-multilabel apply [options] [--step=<step>] <model.pt> <database.task.protocol> <output_dir>
  pyannote-multilabel -h | --help
  pyannote-multilabel --version

Common options: 
  <database.task.protocol>   Experimental protocol (e.g. "BabyTrain.SpeakerRole.JSALT")
  --database=<database.yml>        Path to database configuration file.
  --subset=<subset>          Set subset (train|developement|test).
                             Defaults to "train" in "train" mode. Defaults to
                             "development" in "validate" mode. Defaults to all subsets in
                             "apply" mode.
  --gpu                      Run on GPUs. Defaults to using CPUs.
  --batch=<size>             Set batch size. Has no effect in "train" mode.
                             [default:  32]
  --from=<epoch>             Start {train|validat}ing at epoch <epoch>. Has no
                             effect in "apply" mode. [default:  0]
  --to=<epochs>              End {train|validat}ing at epoch <epoch>.
                             Defaults to keep going forever.
"train" mode: 
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.

"validation" mode: 
  --every=<epoch>            Validate model every <epoch> epochs [default:  1].
  --chronological            Force validation in chronological order.
  --parallel=<n_jobs>        Process <n_jobs> files in parallel. Defaults to
                             using all CPUs.
  <label>                    Label to predict (KCHI, CHI, FEM, MAL or speech).
                             If options overlap and speech have been activated during training, one
                             can also to validate on the classes (SPEECH, OVERLAP).
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  --precision=<precision>    Target detection precision [default:  0.8].
  --use_der                  Indicates if the Detection Error Rate should be used for validating the mode.
                             Default mode uses precision/recall.

"apply" mode: 
  <model.pt>                 Path to the pretrained model.
  --step=<step>              Sliding window step, in seconds.
                             Defaults to 25% of window duration.

Database configuration file <database.yml>: 
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file: 
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process,
    the neural network architecture, and the task addressed.

    ................... <experiment_dir>/config.yml ...................
    # train the network for segmentation
    # see pyannote.audio.labeling.tasks for more details
    task: 
       name:  Segmentation
       params: 
          duration:  3.2     # sub-sequence duration
          per_epoch:  1      # 1 day of audio per epoch
          batch_size:  32    # number of sub-sequences per batch

    # use precomputed features (see feature extraction tutorial)
    feature_extraction: 
       name:  Precomputed
       params: 
          root_dir:  tutorials/feature-extraction

    # use the StackedRNN architecture.
    # see pyannote.audio.labeling.models for more details
    architecture: 
       name:  StackedRNN
       params: 
         rnn:  LSTM
         recurrent:  [32, 20]
         bidirectional:  True
         linear:  [40, 10]

    # use cyclic learning rate scheduler
    scheduler: 
       name:  CyclicScheduler
       params: 
           learning_rate:  auto
    ...................................................................

"train" mode: 
    This will create the following directory that contains the pre-trained
    neural network weights after each epoch: 

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "validate" mode.

    A bunch of values (loss, learning rate, ...) are sent to and can be
    visualized with tensorboard with the following command: 

        $ tensorboard --logdir=<experiment_dir>

"validate" mode: 
    Use the "validate" mode to run validation in parallel to training.
    "validate" mode will watch the <train_dir> directory, and run validation
    experiments every time a new epoch has ended. This will create the
    following directory that contains validation results: 

        <train_dir>/validate/<database.task.protocol>.<subset>

    You can run multiple "validate" in parallel (e.g. for every subset,
    protocol, task, or database).

    In practice, for each epoch, "validate" mode will look for the peak
    detection threshold that maximizes speech turn coverage, under the
    constraint that purity must be greater than the value provided by the
    "--purity" option. Both values (best threshold and corresponding coverage)
    are sent to tensorboard.

"apply" mode
    Use the "apply" mode to extract segmentation raw scores.
    Resulting files can then be used in the following way: 

    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('<output_dir>')

    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('<database.task.protocol>')
    >>> first_test_file = next(protocol.test())

    >>> from pyannote.audio.signal import Peak
    >>> peak_detection = Peak()

    >>> raw_scores = precomputed(first_test_file)
    >>> homogeneous_segments = peak_detection.apply(raw_scores, dimension=1)
"""

from os.path import dirname, basename
from tqdm import tqdm
from functools import partial
from pathlib import Path
import torch
import numpy as np
import scipy.optimize
from docopt import docopt
import multiprocessing as mp
from .base_labeling import BaseLabeling

from pyannote.database import get_annotated, get_protocol, FileFinder

from pyannote.metrics.detection import DetectionErrorRate, DetectionRecall, DetectionPrecision

from pyannote.audio.labeling.extraction import SequenceLabeling

from pyannote.audio.pipeline import SpeakerActivityDetection \
                             as SpeakerActivityDetectionPipeline

from pyannote.audio.features import Precomputed

from pyannote.core.utils.helper import get_class_by_name
from pyannote.core import Timeline, SlidingWindowFeature


def validate_helper_func(current_file, pipeline=None, precision=None, recall=None, label=None, metric=None):
    reference = current_file[label]
    # pipeline has been initialized with label, so that it can know which class needs to be assessed
    hypothesis = pipeline(current_file)
    uem = get_annotated(current_file)

    if precision is not None:
        p = precision(reference, hypothesis, uem=uem)
        r = recall(reference, hypothesis, uem=uem)
        return p, r
    else:
        return metric(reference, hypothesis, uem=uem)


class Multilabel(BaseLabeling):

    def __init__(self, protocol_name, experiment_dir, db_yml=None, training=False, use_der=False):
        super(BaseLabeling, self).__init__(
            experiment_dir, db_yml=db_yml, training=training)

        self.use_der = use_der

        # task
        Task = get_class_by_name(
            self.config_['task']['name'],
            default_module_name='pyannote.audio.labeling.tasks')

        # Here we need preprocessors for initializing the task
        # since we need to know the labels that must be predicted
        # amongst {KCHI,CHI,FEM,MAL,SPEECH,OVERLAP}
        self.task_ = Task(protocol_name=protocol_name,
                          preprocessors=self.preprocessors_,
                          **self.config_['task'].get('params', {}))
        # architecture
        Architecture = get_class_by_name(
            self.config_['architecture']['name'],
            default_module_name='pyannote.audio.labeling.models')

        params = self.config_['architecture'].get('params', {})
        self.get_model_ = partial(Architecture, **params)

        if hasattr(Architecture, 'get_frame_info'):
            self.frame_info_ = Architecture.get_frame_info(**params)
        else:
            self.frame_info_ = None

        if hasattr(Architecture, 'frame_crop'):
            self.frame_crop_ = Architecture.frame_crop
        else:
            self.frame_crop_ = None

    @classmethod
    def from_train_dir(cls, protocol_name, train_dir, db_yml=None, training=False, use_der=False):
        experiment_dir = dirname(dirname(train_dir))
        app = cls(protocol_name, experiment_dir, db_yml=db_yml, training=training, use_der=use_der)
        app.train_dir_ = train_dir
        return app

    @classmethod
    def from_model_pt(cls, protocol_name, model_pt, db_yml=None, training=False, use_der=False):
        train_dir = dirname(dirname(model_pt))
        app = cls.from_train_dir(protocol_name, train_dir, db_yml=db_yml, training=training, use_der=use_der)
        app.model_pt_ = model_pt
        epoch = int(basename(app.model_pt_)[:-3])
        app.model_ = app.load_model(epoch, train_dir=train_dir)
        return app

    def validate_init(self, protocol_name, subset='development'): 

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        files = getattr(protocol, subset)()

        self.pool_ = mp.Pool(mp.cpu_count())

        # pre-compute features for each validation files
        validation_data = []
        for current_file in tqdm(files, desc='Feature extraction'): 

            # precompute features
            if not isinstance(self.feature_extraction_, Precomputed): 
                current_file['features'] = self.feature_extraction_(
                    current_file)

            if self.label == "SPEECH":
                # all the speakers
                current_file[self.label] = current_file['annotation']
            elif self.label in ["CHI", "FEM", "KCHI", "MAL"]:
                reference = current_file['annotation']
                label_speech = reference.subset([self.label])
                current_file[self.label] = label_speech
            elif self.label == "OVERLAP":
                # build overlap reference
                uri = current_file['uri']
                overlap = Timeline(uri=uri)
                turns = current_file['annotation']
                for track1, track2 in turns.co_iter(turns):
                    if track1 == track2:
                        continue
                    overlap.add(track1[0] & track2[0])
                current_file["OVERLAP"] = overlap.support().to_annotation()
            validation_data.append(current_file)
        return validation_data

    def validate_epoch(self, epoch, protocol_name, subset='development', validation_data=None):
        """
        Validate function given a class which must belongs to ["CHI", "FEM", "KCHI", "MAL", "SPEECH", "OVERLAP"]
        """
        # Name of the class that needs to be validated
        class_name = self.label

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        # compute (and store) SAD scores
        duration = self.task_.duration

        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=.25 * duration, batch_size=self.batch_size,
            device=self.device)

        for current_file in validation_data:
            scores = sequence_labeling(current_file)
            if class_name == "SPEECH" and "SPEECH" not in self.task_.labels_:
                # We sum up all the scores of every speakers
                scores_data = np.sum(scores.data, axis=1).reshape(-1, 1)
            else: 
                # We extract the score of interest
                dimension = self.task_.labels_.index(class_name)
                scores_data = scores.data[:, dimension].reshape(-1, 1)

            current_file[class_name+'_scores'] = SlidingWindowFeature(
                scores_data,
                scores.sliding_window)

        # pipeline
        pipeline = SpeakerActivityDetectionPipeline(label=self.label, use_der=self.use_der)

        lower_alpha = 0.
        upper_alpha = 1.
        best_alpha = .5 * (lower_alpha + upper_alpha)
        best_recall = 0.

        if not self.use_der:
            for _ in range(10):

                current_alpha = .5 * (lower_alpha + upper_alpha)
                pipeline.instantiate({'onset':  current_alpha,
                                      'offset':  current_alpha,
                                      'min_duration_on':  0.,
                                      'min_duration_off':  0.,
                                      'pad_onset':  0.,
                                      'pad_offset':  0.})

                precision = DetectionPrecision(parallel=True)
                recall = DetectionRecall(parallel=True)

                validate = partial(validate_helper_func,
                                   pipeline=pipeline,
                                   precision=precision,
                                   recall=recall,
                                   label=self.label)
                _ = self.pool_.map(validate, validation_data)

                precision = abs(precision)
                recall = abs(recall)

                if precision < self.precision:
                    # precision is not high enough:  try higher thresholds
                    lower_alpha = current_alpha

                else:
                    upper_alpha = current_alpha
                    if recall > best_recall:
                        best_recall = recall
                        best_alpha = current_alpha

            return {'metric':  f'recall@{self.precision: .2f}precision',
                    'minimize':  False,
                    'value':  best_recall,
                    'pipeline':  pipeline.instantiate({'onset':  best_alpha,
                                                       'offset':  best_alpha,
                                                       'min_duration_on':  0.,
                                                       'min_duration_off':  0.,
                                                       'pad_onset':  0.,
                                                       'pad_offset':  0.})}
        else:
            def fun(threshold):
                pipeline.instantiate({'onset': threshold,
                                      'offset': threshold,
                                      'min_duration_on': 0.,
                                      'min_duration_off': 0.,
                                      'pad_onset': 0.,
                                      'pad_offset': 0.})
                metric = DetectionErrorRate(parallel=True)
                validate = partial(validate_helper_func,
                                   pipeline=pipeline,
                                   label=self.label,
                                   metric=metric)
                _ = self.pool_.map(validate, validation_data)

                return abs(metric)

            res = scipy.optimize.minimize_scalar(
                fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10})

            threshold = res.x.item()

            return {'metric': 'detection_error_rate',
                    'minimize': True,
                    'value': res.fun,
                    'pipeline': pipeline.instantiate({'onset': threshold,
                                                      'offset': threshold,
                                                      'min_duration_on': 0.,
                                                      'min_duration_off': 0.,
                                                      'pad_onset': 0.,
                                                      'pad_offset': 0.})}

    def apply(self, protocol_name, output_dir, step=None, subset=None): 

        model = self.model_.to(self.device)
        model.eval()

        duration = self.task_.duration
        if step is None: 
            step = 0.25 * duration

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed): 
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, batch_size=self.batch_size,
            device=self.device)

        sliding_window = sequence_labeling.sliding_window
        n_classes = self.task_.n_classes
        labels = self.task_.labels_

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sliding_window,
            dimension=n_classes,
            labels=labels)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        if subset is None: 
            files = FileFinder.protocol_file_iter(protocol,
                                                  extra_keys=['audio'])
        else: 
            files = getattr(protocol, subset)()

        for current_file in files: 
            fX = sequence_labeling(current_file)
            precomputed.dump(current_file, fX)


def main():
    arguments = docopt(__doc__, version='Multilabel')
    db_yml = arguments['--database']
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']
    gpu = arguments['--gpu']
    device = torch.device('cuda') if gpu else torch.device('cpu')
    use_der = arguments['--use_der']

    # HACK for JHU/CLSP cluster
    _ = torch.Tensor([0]).to(device)
    
    if arguments['train']:
        experiment_dir = Path(arguments['<experiment_dir>'])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        if subset is None: 
            subset = 'train'

        # start training at this epoch (defaults to 0)
        restart = int(arguments['--from'])

        # stop training at this epoch (defaults to never stop)
        epochs = arguments['--to']
        if epochs is None: 
            epochs = np.inf
        else: 
            epochs = int(epochs)

        application = Multilabel(protocol_name, experiment_dir, db_yml=db_yml,
                                 training=True)
        application.device = device
        application.train(protocol_name, subset=subset,
                          restart=restart, epochs=epochs)

    if arguments['validate']: 
        label = arguments['<label>']
        precision = float(arguments['--precision'])
        train_dir = Path(arguments['<train_dir>'])
        train_dir = train_dir.expanduser().resolve(strict=True)

        if subset is None: 
            subset = 'development'

        # start validating at this epoch (defaults to 0)
        start = int(arguments['--from'])

        # stop validating at this epoch (defaults to np.inf)
        end = arguments['--to']
        if end is None: 
            end = np.inf
        else: 
            end = int(end)

        # validate every that many epochs (defaults to 1)
        every = int(arguments['--every'])

        # validate epochs in chronological order
        in_order = arguments['--chronological']

        # batch size
        batch_size = int(arguments['--batch'])

        # number of processes
        n_jobs = arguments['--parallel']
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        else:
            n_jobs = int(n_jobs)

        application = Multilabel.from_train_dir(protocol_name, train_dir, db_yml=db_yml, training=False, use_der=use_der)

        application.device = device
        application.batch_size = batch_size
        application.label = label
        application.n_jobs = n_jobs
        application.precision = precision

        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             in_order=in_order, task=label)

    if arguments['apply']: 

        if subset is None: 
            subset = 'test'

        model_pt = Path(arguments['<model.pt>'])
        model_pt = model_pt.expanduser().resolve(strict=True)

        output_dir = Path(arguments['<output_dir>'])
        output_dir = output_dir.expanduser().resolve(strict=False)

        # TODO. create README file in <output_dir>

        step = arguments['--step']
        if step is not None: 
            step = float(step)

        batch_size = int(arguments['--batch'])

        application = Multilabel.from_model_pt(
            protocol_name, model_pt, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
        application.apply(protocol_name, output_dir, step=step, subset=subset)

