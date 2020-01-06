#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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

from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyannote.audio.models.models import RNN


from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature
from pyannote.core import Timeline
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.database import get_annotated
from pyannote.database import get_protocol

from .base import LabelingTask
from .base import LabelingTaskGenerator
from .. import TASK_MULTI_LABEL_CLASSIFICATION
from ..gradient_reversal import GradientReversal


class MultilabelGenerator(LabelingTaskGenerator):
    """Batch generator for training a multi-label classifier

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    labels_spec : `dict`
        Describes the labels that must be predicted.
        1) Must contain a 'regular' key listing the labels appearing 'as-is' in the dataset.
        2) Might contain a 'union' key listing the {key, values} where key is the name of
        the union_label that needs to be predicted, and values is the list of labels
        that will construct the union_label (useful to construct speech classes).
        3) Might contain a 'intersection' key listing the {key, values} where key is the name of
        the intersection_label that needs to be predicted, and values is the list of labels
        that will construct the intersection_label (useful to construct overlap classes).
    subset : {'train', 'development', 'test'}
    frame_info : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
    frame_crop : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models
        that include the feature extraction step (e.g. SincNet) and
        therefore use a different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.

    Usage
    -----
    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/mfcc')

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('BabyTrain.SpeakerRole.JSALT')

    # labels specification
    >>> labels_spec = {'regular': ['CHI', 'FEM', 'MAL'],
    >>>                'union': {
    >>>                     'speech' : ['CHI', 'FEM', 'MAL']
    >>>                     'adult_spech': ['FEM','MAL']
    >>>                 },
    >>>                 'intersection': {
    >>>                     'overlap' : ['CHI', 'FEM', 'MAL']
    >>>                 }
    >>>                }

    # instantiate batch generator
    >>> batches =  MultilabelGenerator(precomputed, protocol, labels_spec)

    # iterate over training set
    >>> for batch in batches(protocol, subset='train'):
    >>>     # batch['X'] is a (batch_size, n_samples, n_features) numpy array
    >>>     # batch['y'] is a (batch_size, n_samples, n_tasks) numpy array
    >>>     pass
    """

    def __init__(self, feature_extraction, protocol, labels_spec,
                 subset='train', frame_info=None, frame_crop=None,
                 duration=3.2, batch_size=32, per_epoch=1, parallel=1,
                 shuffle=True):
        self.labels_spec = labels_spec
        super().__init__(feature_extraction, protocol, subset=subset,
                         frame_info=frame_info, frame_crop=frame_crop,
                         duration=duration,
                         batch_size=batch_size, per_epoch=per_epoch,
                         parallel=parallel, shuffle=shuffle)

    def initialize_y(self, current_file):
        # First, one hot encode the regular classes
        annotation = current_file['annotation'].subset(self.labels_spec['regular'])
        y, _ = one_hot_encoding(annotation,
                                get_annotated(current_file),
                                self.frame_info,
                                labels=self.labels_spec["regular"],
                                mode='center')
        y_data = y.data
        # Then, one hot encode the meta classes
        for derivation_type in ['union', 'intersection']:
            for meta_label, regular_labels in self.labels_spec[derivation_type].items():
                derived = Multilabel.derives_label(current_file["annotation"], derivation_type, meta_label, regular_labels)
                z, _ = one_hot_encoding(derived, get_annotated(current_file),
                                        self.frame_info,
                                        labels=[meta_label],
                                        mode='center')

                y_data = np.hstack((y_data, z.data))

        return SlidingWindowFeature(self.postprocess_y(y_data),
                                    y.sliding_window)

    @property
    def specifications(self):
        specs = {
            'task': TASK_MULTI_LABEL_CLASSIFICATION,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.labels_spec["regular"]
                             + list(self.labels_spec['union'])
                             + list(self.labels_spec['intersection'])},
        }
        for key, classes in self.file_labels_.items():
            specs[key] = {'classes': classes}

        return specs

class Multilabel(LabelingTask):
    """
    Train a n-labels classifier where the labels are provided by the user, and can be of 3 types :

    - Regular labels : those are extracted directly from the annotation and are kept unchanged.
    - Union meta-label : those are extracted by taking the union of multiple regular labels.
    - Intersection meta-label : those are extracted by taking the intersection of multiple regular labels.

    Parameters
    ----------
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    label_spec: `dict`
        regular: list
            List of classes that need to be predicted.
        union:
            Dictionnary of union meta-labels whose keys are the meta-label names,
            and values are a list of regular classes
        intersection:
            Dictionnary of intersection meta-labels whose keys are the meta-label names,
            and values are a list of regular classes
    weighted_loss: 'boolean', optional, default to False
        Indicates if loss should be weighted by 1 / prior

    Usage in config.yml
    --------------------
    task:
        name: Multilabel
        params:
            duration: 2.0         # sequences are 2s long
            batch_size: 16        # 64 sequences per batch
            per_epoch: 1          # one epoch = 1 day of audio
            weighted_loss: True   # Weight loss by 1 / prior
            labels_spec:
                regular: ['CHI', 'MAL', 'FEM']
                union:
                    speech: ['CHI', 'FEM', 'MAL']     # build speech label
                    adult_speech : ['FEM', 'MAL']     # build adult_speech label
                intersection:
                    overlap: ['CHI', 'MAL', 'FEM']    # build overlap label
    Usage
    -----
    # Use mapping as a preprocessor
    >>> from pyannote.database.util import LabelMapper
    >>> preprocessors = {'annotation': LabelMapper(mapping=mapping)}

    # labels specification
    >>> labels_spec = {'regular': ['CHI', 'FEM', 'MAL'],
    >>>                'union': {
    >>>                     'speech' : ['CHI', 'FEM', 'MAL']
    >>>                     'adult_spech': ['FEM','MAL']
    >>>                 },
    >>>                 'intersection': {
    >>>                     'overlap' : ['CHI', 'FEM', 'MAL']
    >>>                 }
    >>>                }

    # protocol name
    >>> protocol_name = 'BabyTrain.SpeakerDiarization.All'
    >>> task = Multilabel(protocol_name, preprocessors, labels_spec)

    # model architecture
    >>> from pyannote.audio.labeling.models import StackedRNN
    >>> model = StackedRNN(task.specifications)

    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/features')

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol(protocol_name)

    # train model using protocol training set
    >>> for epoch, model in task.fit_iter(model, task.get_batch_generator(precomputed, protocol)):
    ...     pass
    """
    def __init__(self, labels_spec, weighted_loss=False, **kwargs):
        super(Multilabel, self).__init__(**kwargs)

        # Labels related attributes
        self.labels_spec = labels_spec
        labels_spec_key = self.labels_spec.keys()
        if 'regular' not in labels_spec_key:
            self.labels_spec['regular'] = dict()
        if 'union' not in labels_spec_key:
            self.labels_spec['union'] = dict()
        if 'intersection' not in labels_spec_key:
            self.labels_spec['intersection'] = dict()

        self.regular_labels = self.labels_spec['regular']
        self.union_labels = list(self.labels_spec['union'])
        self.intersection_labels = list(self.labels_spec['intersection'])

        self.label_names = self.regular_labels +\
                           self.union_labels +\
                           self.intersection_labels

        if set(self.union_labels).intersection(self.intersection_labels):
            raise ValueError("Union keys and intersection keys in "
                             "labels_spec should be mutually exclusive.")

        self.nb_regular_labels = len(labels_spec["regular"])

        self.weight_ = None
        self.weighted_loss = weighted_loss
        self.n_classes_ = self.nb_regular_labels + len(self.union_labels) + len(self.intersection_labels)

    @staticmethod
    def derives_label(annotation, derivation_type, meta_label, regular_labels):
        """Returns an Annotation describing the utterances of the union or intersection
        of multiple labels.

        Derives a label. The derivation takes as inputs :
        - An annotation from which we want to derive
        - A derivation type : union or intersection
        - A meta label : the name of the output label
        - A list of regular labels : the regular labels from which we want to derive

        Example :
            1) derives_label(annotation, 'union', 'speech', ["CHI","MAL","FEM"]
            Will compute the speech label based on the union of "CHI", "MAL" and "FEM"
            2) derives_label(annotation, 'intersection', 'overlap', ["CHI","MAL","FEM"]
            Will compute the overlapping speech based on the intersection of "CHI", "MAL" and "FEM"

        Parameters
        ----------
        annotation : `Annotation`
            Input annotation that needs to be derived
        derivation_type: {'union', 'intersection'}
            Indicates if the union, or the intersection must be considered
        meta_label: `string`
            Indicates the name of the output label
        regular_labels: `list`
            Indicates the list of labels that must be taken into account.

        Returns
        -------
        variable_name : `Annotation`
            Annotation whose only label is meta_label that has been constructed
            by taking the intersection or union of the regular_labels list.

        Usage
        -----
        # compute the "adult_speech" label
        >>> speech = derives_label(annotation, 'union', 'adult_speech', ['MAL', 'FEM'])
        # compute the "overlap" label blahlblah
        >>> overlap = derives_label(annotation, 'intersection', 'overlap', ['MAL','FEM','CHI'])
        """

        if derivation_type not in ['union', 'intersection']:
            raise ValueError("Derivation type must be in ['union', 'intersection']")

        derived = Annotation(uri=annotation.uri)
        mapping = {k: meta_label for k in regular_labels}
        annotation = annotation.subset(regular_labels).rename_labels(mapping=mapping)

        if derivation_type == 'union':
            support = annotation.support()
            return derived.update(support)
        elif derivation_type == 'intersection':
            overlap = Timeline()
            for track1, track2 in annotation.co_iter(annotation):
                if track1 == track2:
                    continue
                overlap.add(track1[0] & track2[0])
            derived = overlap.support().to_annotation(generator=cycle([meta_label]))
            return derived
        else:
            raise ValueError("derivation_type must belong to ['union', 'intersection']\n"
                             "Can't be %s." % derivation_type)

    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            frame_info=None, frame_crop=None):
        return MultilabelGenerator(
            feature_extraction,
            protocol, subset=subset,
            frame_info=frame_info,
            frame_crop=frame_crop,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            parallel=self.parallel,
            labels_spec=self.labels_spec)

    @property
    def weight(self):
        return self.weight_

    def on_train_start(self):
        if self.weighted_loss:
            weights = dict([(key, 0.0) for key in self.label_names[0:self.nb_regular_labels]])
            for file in self.batch_generator_.data_.values():
                y = file['current_file']['annotation']
                for speaker in self.regular_labels[0:self.nb_regular_labels]:
                    weights[speaker] += y.label_duration(speaker)

            # Taking the inverse of prior
            total_speech = sum(weights.values(), 0.0)
            for key, value in weights.items():
                if value != 0:
                    weights[key] = total_speech / value

            # Finally, normalize so that the weights sum to 1
            norm1 = sum(weights.values())
            regular_weights = {key: value / norm1 for key, value in weights.items()}
            meta_weights = {key: 1 for key in self.union_labels + self.intersection_labels}

            weights = list(regular_weights.values()) + list(meta_weights.values())
            self.weight_ = torch.tensor(np.array(weights), dtype=torch.float32)

        self.task_type_ = self.model_.specifications['task']

        def loss_func(input, target, weight=None, mask=None):
            print(mask)
            if mask is None:
                return F.binary_cross_entropy(input, target, weight=weight,
                                              reduction='mean')
            else:
                return torch.mean(
                    mask * F.binary_cross_entropy(input, target,
                                                  weight=weight,
                                                  reduction='none'))

        self.loss_func_ = loss_func


class DomainAwareMultilabel(Multilabel):
    """Domain-aware multilabel classification

    Trains multilabel and domain classification jointly.

    Parameters
    ----------
    domain : `str`, optional
        Batch key to use as domain. Defaults to 'domain'.
        Could be 'database' or 'uri' for instance.
    attachment : `int`, optional
        Intermediate level where to attach the domain classifier.
        Defaults to -1. Passed to `return_intermediate` in models supporting it.
    rnn : `dict`, optional
        Parameters of the RNN used in the domain classifier.
        See `pyannote.audio.models.models.RNN` for details.
    domain_loss : `str`, optional
        Loss function to use. Defaults to 'NLLLoss'.
    """

    DOMAIN_PT = '{log_dir}/weights/{epoch:04d}.domain.pt'

    def __init__(self,
                 domain='domain', attachment=-1,
                 rnn=None, domain_loss="NLLLoss",
                 **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self.attachment = attachment

        if rnn is None:
            rnn = dict()
        self.rnn = rnn

        self.domain_loss = domain_loss
        if self.domain_loss == "NLLLoss":
            # Default value
            self.domain_loss_ = nn.NLLLoss()
            self.activation_ = nn.LogSoftmax(dim=1)

        elif self.domain_loss == "MSELoss":
            self.domain_loss_ = nn.MSELoss()
            self.activation_ = nn.Sigmoid()

        else:
            msg = (
                f'{domain_loss} has not been implemented yet.'
            )
            raise NotImplementedError(msg)

    def parameters(self, model, specifications, device):
        """Initialize trainable trainer parameters

        Parameters
        ----------
        specifications : `dict`
            Batch specs.

        Returns
        -------
        parameters : iterable
            Trainable trainer parameters
        """
        domain_classifier_rnn = RNN(
            n_features=model.intermediate_dimension(self.attachment),
            **self.rnn)
        print(domain_classifier_rnn)
        domain_classifier_linear = nn.Linear(
            domain_classifier_rnn.dimension,
            len(specifications[self.domain]['classes']),
            bias=True).to(device)

        self.domain_classifier_ = nn.Sequential(domain_classifier_rnn,
                                                domain_classifier_linear).to(device)
        return list(self.domain_classifier_.parameters())

    def load_epoch(self, epoch):
        """Load model and classifier from disk

        Parameters
        ----------
        epoch : `int`
            Epoch number.
        """

        super().load_epoch(epoch)

        domain_classifier_state = torch.load(
            self.DOMAIN_PT.format(log_dir=self.log_dir_, epoch=epoch),
            map_location=lambda storage, loc: storage)
        self.domain_classifier_.load_state_dict(domain_classifier_state)

    def save_epoch(self, epoch=None):
        """Save model to disk

        Parameters
        ----------
        epoch : `int`, optional
            Epoch number. Defaults to self.epoch_

        """

        if epoch is None:
            epoch = self.epoch_

        torch.save(self.domain_classifier_.state_dict(),
                   self.DOMAIN_PT.format(log_dir=self.log_dir_,
                                         epoch=epoch))

        super().save_epoch(epoch=epoch)

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """

        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device=self.device_)
        fX, intermediate = self.model_(X, return_intermediate=self.attachment)

        # speech activity detection
        fX = fX.view((-1, self.n_classes_))
        target = torch.tensor(
            batch['y'],
            dtype=torch.int64,
            device=self.device_).contiguous().view((-1,))

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=self.device_)
        loss = self.loss_func_(fX, target, weight=weight)

        # domain classification
        domain_target = torch.tensor(
            batch[self.domain],
            dtype=torch.int64,
            device=self.device_)

        domain_scores = self.activation_(self.domain_classifier_(intermediate))

        domain_loss = self.domain_loss_(domain_scores, domain_target)

        return {'loss': loss + domain_loss,
                'loss_domain': domain_loss,
                'loss_task': loss}


class DomainAdversarialMultilabel(DomainAwareMultilabel):
    """Domain Adversarial speech activity detection

    Parameters
    ----------
    domain : `str`, optional
        Batch key to use as domain. Defaults to 'domain'.
        Could be 'database' or 'uri' for instance.
    attachment : `int`, optional
        Intermediate level where to attach the domain classifier.
        Defaults to -1. Passed to `return_intermediate` in models supporting it.
    alpha : `float`, optional
        Coefficient multiplied with the domain loss
    """

    def __init__(self, domain='domain', attachment=-1, alpha=1., **kwargs):
        super().__init__(domain=domain, attachment=attachment, **kwargs)
        self.alpha = alpha
        self.gradient_reversal_ = GradientReversal()

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """
        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device=self.device_)

        fX, intermediate = self.model_(X, return_intermediate=self.attachment)

        # multilabel classification
        target = torch.tensor(
            batch['y'],
            dtype=torch.float32,
            device=self.device_)

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=self.device_)

        loss = self.loss_func_(fX, target, weight=weight)

        # domain classification
        domain_target = torch.tensor(
            batch[self.domain],
            dtype=torch.int64,
            device=self.device_)

        domain_scores = self.activation_(self.domain_classifier_(
            self.gradient_reversal_(intermediate)))

        if self.domain_loss == "MSELoss":
            # One hot encode domain_target for Mean Squared Error Loss
            nb_domains = domain_scores.shape[1]
            identity_mat = torch.sparse.torch.eye(nb_domains, device=self.device_)
            domain_target = identity_mat.index_select(dim=0, index=domain_target)

        domain_loss = self.domain_loss_(domain_scores, domain_target)

        return {'loss': loss + self.alpha * domain_loss,
                'loss_domain': domain_loss,
                'loss_task': loss}
