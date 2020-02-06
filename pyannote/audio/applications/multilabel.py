#!/usr/bin/env python
# encoding:  utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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
Multi-label classifier

Usage: 
  pyannote-multilabel train [options] <experiment_dir> <database.task.protocol>
  pyannote-multilabel validate [options] [--every=<epoch> --chronological --precision=<precision> --detection] <label> <train_dir> <database.task.protocol>
  pyannote-multilabel apply [options] [--detection] [--step=<step>] <validate_dir> <database.task.protocol>
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
  --detection                Indicates if the Detection Error Rate should be used for validating mode.
                             Default mode uses precision/recall.
  --fscore                   Indicates if the precision/recall F measure should be used for both validating and application mode.
                             Default mode uses precision/recall.
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
  <label>                    Label that needs to be validated. Must belong to the labels
                             that have been seen during training.
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  --precision=<precision>    Target detection precision [default:  0.8].

"apply" mode:
  <validate_dir>             Path to the directory containing validation
                             results (i.e. the output of "validate" mode).
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
    # train the network for the multilabel task
    # see pyannote.audio.labeling.tasks for more details
    task:
        name: Multilabel
        params:
            duration: 2.0     # sequences are 2s long
            batch_size: 16     # 64 sequences per batch
            per_epoch: 1       # one epoch = 1 day of audio
            labels_spec:
                regular: ['CHI', 'MAL', 'FEM']
                union:
                    speech: ['CHI', 'FEM', 'MAL']   # build speech label
                    adult_speech : ['FEM', 'MAL']   # build adult_speech label
                intersection:
                    overlap : ['CHI', 'MAL', 'FEM'] # build overlap label


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

    # Label mapping : depends on the labels found in your data
    preprocessors:
        annotation:
           name: pyannote.database.util.LabelMapper
           params:
             keep_missing: False    # Raise an error if one of the input label is not found in the mapping.
             mapping:
                "BRO": "CHI"
                "MOT": "FEM"
                "FAT": "MAL"
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

        <train_dir>/validate_<label>/<database.task.protocol>.<subset>

    You can run multiple "validate" in parallel (e.g. for every subset,
    protocol, task, database or label).

    In practice, for each epoch, "validate" mode will look for the optimal
    decision threshold that maximizes recall, depending on a given accuracy.
    If --detection mode is activated, it will minimizes the detection error rate instead.

"apply" mode:
    Use the "apply" mode to extract speech activity detection raw scores and
    results. This will create the following directory that contains speech
    activity detection results:
        <validate_dir>/apply/<epoch>
"""

import multiprocessing as mp
from functools import partial
from typing import Optional
from pathlib import Path

import numpy as np
import scipy.optimize
import torch
from docopt import docopt
from pyannote.audio.features import Precomputed
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.labeling.tasks import Multilabel as MultilabelTask
from pyannote.audio.pipeline import SpeechActivityDetection \
    as SpeechActivityDetectionPipeline
from pyannote.core import SlidingWindowFeature
from pyannote.database import get_annotated, get_protocol, FileFinder
from pyannote.metrics.detection import DetectionErrorRate, DetectionRecall, DetectionPrecision, DetectionPrecisionRecallFMeasure

from .base_labeling import BaseLabeling


def validate_helper_func(current_file, pipeline=None, precision=None, recall=None, metric=None):
    reference = current_file["annotation"]
    scores = pipeline(current_file)
    uem = get_annotated(current_file)

    if precision is not None:
        p = precision(reference, scores, uem=uem)
        r = recall(reference, scores, uem=uem)
        return p, r
    else:
        return metric(reference, scores, uem=uem)


class Multilabel(BaseLabeling):

    def validate_init(self, protocol_name, subset='development'):
        validation_data = super().validate_init(protocol_name, subset=subset)

        if self.label in self.task_.labels_spec["regular"]:
            derivation_type = "regular"
        elif self.label in self.task_.labels_spec["union"]:
            derivation_type = "union"
        elif self.label in self.task_.labels_spec["intersection"]:
            derivation_type = "intersection"
        else:
            raise ValueError("%s not found in training labels : %s"
                             % (self.label, self.task_.label_names))

        # Overwrite annotation field with the class of interest
        for current_file in validation_data:
            if derivation_type == "regular":
                current_file["annotation"] = current_file["annotation"].subset([self.label])
            else:
                current_file["annotation"] = MultilabelTask.derives_label(current_file["annotation"],
                                                                          derivation_type=derivation_type,
                                                                          meta_label=self.label,
                                                                          regular_labels=self.task_.labels_spec[derivation_type][self.label])

        return validation_data

    def validate_epoch(self, epoch, protocol_name, subset='development', validation_data=None):
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

            # We extract the score of interest
            dimension = self.task_.label_names.index(class_name)
            scores_data = scores.data[:, dimension].reshape(-1, 1)
            current_file['scores'] = SlidingWindowFeature(
                scores_data,
                scores.sliding_window)

        # pipeline
        pipeline = SpeechActivityDetectionPipeline(scores_name='scores',
                                                   detection=self.detection)

        lower_alpha = 0.
        upper_alpha = 1.
        best_alpha = .5 * (lower_alpha + upper_alpha)
        best_recall = 0.

        if not self.detection and not self.fscore:
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
                                   recall=recall)
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
        elif self.detection:
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
        elif self.fscore:

            def fun(threshold):
                pipeline.instantiate({'onset': threshold,
                                      'offset': threshold,
                                      'min_duration_on': 0.,
                                      'min_duration_off': 0.,
                                      'pad_onset': 0.,
                                      'pad_offset': 0.})
                metric = DetectionPrecisionRecallFMeasure(parallel=True)
                validate = partial(validate_helper_func,
                                   pipeline=pipeline,
                                   metric=metric)
                _ = self.pool_.map(validate, validation_data)
                return -abs(metric)

            res = scipy.optimize.minimize_scalar(
                fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10})

            threshold = res.x.item()

            return {'metric': 'fscore',
                    'minimize': False,
                    'value': -res.fun,
                    'pipeline': pipeline.instantiate({'onset': threshold,
                                                      'offset': threshold,
                                                      'min_duration_on': 0.,
                                                      'min_duration_off': 0.,
                                                      'pad_onset': 0.,
                                                      'pad_offset': 0.})}

    def apply(self, protocol_name: str,
              step: Optional[float] = None,
              subset: Optional[str] = "test"):

        model = self.model_.to(self.device)
        model.eval()

        labels = model.specifications['y']['classes']
        predicted_class = self.validate_dir_.parent.name.split('_')[-1]
        index_predicted = labels.index(predicted_class)

        if predicted_class in self.task_.labels_spec["regular"]:
            derivation_type = "regular"
        elif predicted_class in self.task_.labels_spec["union"]:
            derivation_type = "union"
        elif predicted_class in self.task_.labels_spec["intersection"]:
            derivation_type = "intersection"
        else:
            raise ValueError("%s not found in training labels : %s"
                             % (self.label, self.task_.label_names))


        duration = self.task_.duration
        if step is None:
            step = 0.25 * duration

        output_dir = Path(self.APPLY_DIR.format(
            validate_dir=self.validate_dir_,
            epoch=self.epoch_))

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, batch_size=self.batch_size,
            device=self.device)

        sliding_window = sequence_labeling.sliding_window

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sliding_window,
            labels=model.classes)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for current_file in getattr(protocol, subset)():
            fX = sequence_labeling(current_file)
            precomputed.dump(current_file, fX)

        # do not proceed with the full pipeline
        # when there is no such thing for current task
        if not hasattr(self, 'pipeline_params_'):
            return

        # instantiate pipeline
        pipeline = SpeechActivityDetectionPipeline(scores=output_dir, dimension=index_predicted)
        pipeline.detection = self.detection
        pipeline.fscore = self.fscore
        pipeline.instantiate(self.pipeline_params_)

        # load pipeline metric (when available)
        try:
            metrics = [pipeline.get_metric()]
        except NotImplementedError as e:
            metrics = [DetectionPrecision(), DetectionRecall()]

        # apply pipeline and dump output to RTTM files
        output_rttm = output_dir / f'{protocol_name}.{subset}.rttm'
        with open(output_rttm, 'w') as fp:
            for current_file in getattr(protocol, subset)():
                hypothesis = pipeline(current_file)
                pipeline.write_rttm(fp, hypothesis)

                # compute evaluation metric (when possible)
                if 'annotation' not in current_file:
                    metrics = None

                # compute evaluation metric (when available)
                if metrics is None:
                    continue

                if derivation_type == "regular":
                    current_file["annotation"] = current_file["annotation"].subset([predicted_class])
                else:
                    current_file["annotation"] = MultilabelTask.derives_label(current_file["annotation"],
                                                                              derivation_type=derivation_type,
                                                                              meta_label=predicted_class,
                                                                              regular_labels=
                                                                              self.task_.labels_spec[derivation_type][predicted_class])
                reference = current_file['annotation']
                uem = get_annotated(current_file)
                for metric in metrics:
                    _ = metric(reference, hypothesis, uem=uem)

        for metric in metrics:
            name = metric.metric_name().replace(' ', '_')
            output_eval = output_dir / f'{protocol_name}.{subset}.{name}.eval'
            with open(output_eval, 'w') as fp:
                fp.write(str(metric))

def main():
    arguments = docopt(__doc__, version='Multilabel')
    db_yml = arguments['--database']
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']
    gpu = arguments['--gpu']
    device = torch.device('cuda') if gpu else torch.device('cpu')
    detection = arguments['--detection']
    fscore = arguments['--fscore']

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

        application = Multilabel(experiment_dir, db_yml=db_yml,
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

        application = Multilabel.from_train_dir(train_dir, db_yml=db_yml, training=False)

        application.device = device
        application.batch_size = batch_size
        application.label = label
        application.n_jobs = n_jobs
        application.precision = precision
        application.detection = detection
        application.fscore = fscore

        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             in_order=in_order, task=label)

    if arguments['apply']:
        validate_dir = Path(arguments['<validate_dir>'])
        validate_dir = validate_dir.expanduser().resolve(strict=True)

        if subset is None: 
            subset = 'test'

        step = arguments['--step']
        if step is not None: 
            step = float(step)

        batch_size = int(arguments['--batch'])

        application = Multilabel.from_validate_dir(
            validate_dir, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
        application.detection = detection
        application.fscore = fscore
        application.apply(protocol_name, step=step, subset=subset)

