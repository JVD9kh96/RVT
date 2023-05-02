from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
from typing import List

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_rnd import SequenceForRandomAccess
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DatasetMode, LoaderDataDictGenX, DatasetType, DataType


import numpy as np
import tensorflow as tf

class SequenceDataset(tf.keras.utils.Sequence):
    def __init__(self,
                 path: Path,
                 dataset_mode: str,
                 dataset_config: DictConfig,
                 batch_size: int):
        assert path.is_dir()

        ### extract settings from config ###
        sequence_length = dataset_config.sequence_length
        assert isinstance(sequence_length, int)
        assert sequence_length > 0
        self.output_seq_len = sequence_length

        ev_representation_name = dataset_config.ev_repr_name
        downsample_by_factor_2 = dataset_config.downsample_by_factor_2
        only_load_end_labels = dataset_config.only_load_end_labels

        augm_config = dataset_config.data_augmentation

        ####################################
        if dataset_config.name == 'gen1':
            dataset_type = DatasetType.GEN1
        elif dataset_config.name == 'gen4':
            dataset_type = DatasetType.GEN4
        else:
            raise NotImplementedError
        self.sequence = SequenceForRandomAccess(path=path,
                                                ev_representation_name=ev_representation_name,
                                                sequence_length=sequence_length,
                                                dataset_type=dataset_type,
                                                downsample_by_factor_2=downsample_by_factor_2,
                                                only_load_end_labels=only_load_end_labels)

        self.spatial_augmentor = None
        if dataset_mode == 'train':
            resolution_hw = tuple(dataset_config.resolution_hw)
            assert len(resolution_hw) == 2
            ds_by_factor_2 = dataset_config.downsample_by_factor_2
            if ds_by_factor_2:
                resolution_hw = tuple(x // 2 for x in resolution_hw)
            self.spatial_augmentor = RandomSpatialAugmentorGenX(
                dataset_hw=resolution_hw,
                automatic_randomization=True,
                augm_config=augm_config.random)

        self.batch_size = batch_size

    def __len__(self):
        return len(self.sequence) // self.batch_size

    def on_epoch_end(self):
        self.sequence.shuffle()

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        items = self.sequence[start_idx:end_idx]

        if self.spatial_augmentor is not None and not self.sequence.is_only_loading_labels():
            items = [self.spatial_augmentor(item) for item in items]

        return items


class CustomConcatDataset(tf.keras.utils.Sequence):
    def __init__(self, datasets: Iterable[SequenceDataset]):
        self.datasets = datasets

        self.num_samples = 0
        self.cumulative_sizes = [0]
        for dataset in self.datasets:
            self.num_samples += len(dataset)
            self.cumulative_sizes.append(self.num_samples)

    def __len__(self):
        return self.num_samples

    def on_epoch_end(self):
        for dataset in self.datasets:
            dataset.on_epoch_end()

    def __getitem__(self, index: int):
        dataset_idx = np.searchsorted(self.cumulative_sizes, index, side='right') - 1
        sample_idx = index - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]


def build_random_access_dataset(dataset_mode: DatasetMode, dataset_config: DictConfig) -> CustomConcatDataset:
    dataset_path = Path(dataset_config.path)
    assert dataset_path.is_dir(), f'{str(dataset_path)}'

    mode2str = {DatasetMode.TRAIN: 'train',
                DatasetMode.VALIDATION: 'val',
                DatasetMode.TESTING: 'test'}

    split_path = dataset_path / mode2str[dataset_mode]
    assert split_path.is_dir()

    seq_datasets = list()
    for entry in tqdm(split_path.iterdir(), desc=f'creating rnd access {mode2str[dataset_mode]} datasets'):
        seq_datasets.append(SequenceDataset(path=entry, dataset_mode=dataset_mode, dataset_config=dataset_config))

    return CustomConcatDataset(seq_datasets)


def get_weighted_random_sampler(dataset):
    class2count = {}
    classandcount_list = []
    print('--- START generating weighted random sampler ---')
    dataset.only_load_labels()
    for idx, data in enumerate(tf.data.Dataset.as_numpy_iterator(dataset)):
        labels = data[DataType.OBJLABELS_SEQ]
        label_list, valid_batch_indices = labels.get_valid_labels_and_batch_indices()
        class_ids_seq = []
        for label in label_list:
            class_ids_numpy = np.asarray(label.class_id.numpy(), dtype='int32')
            class_ids_seq.append(class_ids_numpy)
        class_ids_seq, counts_seq = np.unique(np.concatenate(class_ids_seq), return_counts=True)
        for class_id, count in zip(class_ids_seq, counts_seq):
            class2count[class_id] = class2count.get(class_id, 0) + count
        classandcount_list.append((class_ids_seq, counts_seq))
    dataset.load_everything()

    class2weight = {}
    for class_id, count in class2count.items():
        count = max(count, 1)
        class2weight[class_id] = 1 / count

    weights = []
    for classandcount in classandcount_list:
        weight = 0
        for class_id, count in zip(classandcount[0], classandcount[1]):
            # Not only weight depending on class but also depending on number of occurrences.
            # This will bias towards sampling "frames" with more bounding boxes.
            weight += class2weight[class_id] * count
        weights.append(weight)

    print('--- DONE generating weighted random sampler ---')
    return tf.data.experimental.rejection_resample(dataset, weights=weights, target_prob=1/len(weights), seed=None, reshuffle_each_iteration=True, window_size=None, variable_window_size=False, name=None)    
