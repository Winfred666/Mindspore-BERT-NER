# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Data operations, will be used in run_pretrain.py
"""
import os
import math
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
from mindspore import log as logger


class BucketDatasetGenerator:
    """
    Provide data distribution of different gears for the bert network.

    Args:
        dataset (Dataset): The training dataset.
        batch_size (Int): The training batchsize.
        bucket_list (List): List of different sentence lengths, such as [128, 256, 512]. Default: None.
    """

    def __init__(self, dataset, batch_size, bucket_list=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_list = bucket_list
        bucket_size = len(bucket_list)
        self.random_list = np.random.binomial(n=(bucket_size - 1), p=0.55, size=self.__len__())
        self.random_list = (self.random_list + 2) % bucket_size
        self.random_list = [bucket_list[i] for i in self.random_list]
        self._init_variables()

    def _init_variables(self):
        self.data_bucket = {bucket: [] for bucket in self.bucket_list}
        self.iter = 0
        self.remaining_data = []
        self.stage = 0

    def __next__(self):
        if self.stage != 0:
            return self._process_remaining_data()

        for item in self.iterator:
            for seq_length in self.bucket_list:
                if np.sum(item[1]) <= seq_length:
                    self.data_bucket[seq_length].append(item)
                    break
            for key in self.data_bucket.keys():
                data = self.data_bucket[key]
                if len(data) >= self.batch_size and self.random_list[self.iter] == key:
                    self.data_bucket[key] = self.data_bucket[key][self.batch_size:]
                    self.iter += 1
                    return self._package_data(data, key)
        self.stage = 1
        return self._process_remaining_data()

    def _package_data(self, data, key):
        """package a set of data."""
        arr = data[0]
        for i in range(1, self.batch_size):
            current_data = data[i]
            for j in range(len(current_data)):
                arr[j] = np.concatenate((arr[j], current_data[j]))
        res = ()
        for label in arr:
            newlabel = np.reshape(label, (self.batch_size, -1))
            res += (newlabel,)
        res += (np.array(key, np.int32),)
        return res

    def _process_remaining_data(self):
        """process remaining data."""
        for key in self.data_bucket.keys():
            data = self.data_bucket[key]
            if len(data) >= self.batch_size:
                self.data_bucket[key] = self.data_bucket[key][self.batch_size:]
                self.iter += 1
                return self._package_data(data, key)

        for value in self.data_bucket.values():
            self.remaining_data += list(value)
        self.data_bucket = dict()

        if self.batch_size > len(self.remaining_data) or self.iter >= self.__len__():
            self._init_variables()
            raise StopIteration

        remaining_data = self.remaining_data[:self.batch_size]
        self.remaining_data = self.remaining_data[self.batch_size:]
        self.iter += 1
        return self._package_data(remaining_data, self.bucket_list[-1])

    def __iter__(self):
        self._init_variables()
        self.iterator = self.dataset.create_tuple_iterator(output_numpy=True)
        return self

    def __len__(self):
        return self.dataset.get_dataset_size() // self.batch_size


def create_bert_dataset(device_num=1, rank=0, do_shuffle="true", data_dir=None, schema_dir=None, batch_size=32,
                        bucket_list=None, dataset_format="mindrecord", num_samples=None):
    """create train dataset"""
    # apply repeat operations
    files = os.listdir(data_dir)
    data_files = []
    for file_name in files:
        if (dataset_format == "tfrecord" and "tfrecord" in file_name) or \
                (dataset_format == "mindrecord" and "mindrecord" in file_name and "mindrecord.db" not in file_name):
            data_files.append(os.path.join(data_dir, file_name))
    if dataset_format == "mindrecord":
        if str(num_samples).lower() != "none":
            data_set = ds.MindDataset(data_files,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                      shuffle=False, num_shards=device_num, shard_id=rank, num_samples=num_samples)
        else:
            data_set = ds.MindDataset(data_files,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                      shuffle=ds.Shuffle.FILES if do_shuffle == "true" else False,
                                      num_shards=device_num, shard_id=rank)
    elif dataset_format == "tfrecord":
        data_set = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                      shuffle=ds.Shuffle.FILES if do_shuffle == "true" else False,
                                      num_shards=device_num, shard_id=rank, shard_equal_rows=True)
    else:
        raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")
    if bucket_list:
        bucket_dataset = BucketDatasetGenerator(data_set, batch_size, bucket_list=bucket_list)
        data_set = ds.GeneratorDataset(bucket_dataset,
                                       column_names=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                     "masked_lm_positions", "masked_lm_ids", "masked_lm_weights",
                                                     "sentence_flag"],
                                       shuffle=False)
    else:
        data_set = data_set.batch(batch_size, drop_remainder=True)
    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
    data_set = data_set.map(operations=type_cast_op, input_columns="next_sentence_labels")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set


def extract_real_len(array):
    return array[0]  # Extract the first (and only) element of the array.

def create_ner_dataset(batch_size=1, assessment_method="accuracy", data_file_path=None,
                       dataset_format="mindrecord", schema_file_path=None, do_shuffle=True, drop_remainder=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    # loading the mindrecord dataset into memory !!!
    if dataset_format == "mindrecord":
        dataset = ds.MindDataset([data_file_path],
                                 columns_list=["input_ids", "input_mask", "segment_ids", "label_ids","real_seq_length"],
                                 shuffle=do_shuffle)
    elif dataset_format == "tfrecord":
        dataset = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                     columns_list=["input_ids", "input_mask", "segment_ids", "label_ids","real_seq_length"],
                                     shuffle=do_shuffle)
    else:
        raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")
    
    # if using different evalidate assessment, should process differently.
    if assessment_method == "Spearman_correlation":
        type_cast_op_float = C.TypeCast(mstype.float32)
        dataset = dataset.map(operations=type_cast_op_float, input_columns="label_ids")
    else:
        dataset = dataset.map(operations=type_cast_op, input_columns="label_ids")
    
    # this is casting type to int32, suitale to feed into model.
    dataset = dataset.map(operations=type_cast_op, input_columns="segment_ids")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_mask")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_ids")
    
    # extract real_len out of [real_len] array.
    dataset = dataset.map(operations=type_cast_op, input_columns="real_seq_length")
    dataset = dataset.map(operations=extract_real_len, input_columns="real_seq_length")

    # apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset


def create_classification_dataset(batch_size=1, assessment_method="accuracy", data_file_path=None,
                                  schema_file_path=None, dataset_format="mindrecord", do_shuffle=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    if dataset_format == "mindrecord":
        data_set = ds.MindDataset([data_file_path],
                                  columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                  shuffle=do_shuffle)
    elif dataset_format == "tfrecord":
        data_set = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
                                      shuffle=do_shuffle)
    else:
        raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")
    if assessment_method == "Spearman_correlation":
        type_cast_op_float = C.TypeCast(mstype.float32)
        data_set = data_set.map(operations=type_cast_op_float, input_columns="label_ids")
    else:
        data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


def generator_squad(data_features):
    for feature in data_features:
        yield (feature.input_ids, feature.input_mask, feature.segment_ids, feature.unique_id)


def create_squad_dataset(batch_size=1, data_file_path=None, schema_file_path=None,
                         is_training=True, do_shuffle=True, dataset_format="mindrecord"):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    if is_training:
        if dataset_format == "mindrecord":
            data_set = ds.MindDataset([data_file_path],
                                      columns_list=["input_ids", "input_mask", "segment_ids", "start_positions",
                                                    "end_positions", "unique_ids", "is_impossible"],
                                      shuffle=do_shuffle)
        elif dataset_format == "tfrecord":
            data_set = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                                          columns_list=["input_ids", "input_mask", "segment_ids", "start_positions",
                                                        "end_positions", "unique_ids", "is_impossible"],
                                          shuffle=do_shuffle)
        else:
            raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")
        data_set = data_set.map(operations=type_cast_op, input_columns="start_positions")
        data_set = data_set.map(operations=type_cast_op, input_columns="end_positions")
    else:
        if dataset_format == "mindrecord":
            data_set = ds.MindDataset([data_file_path],
                                      columns_list=["input_ids", "input_mask", "segment_ids", "unique_ids"],
                                      shuffle=do_shuffle)
        elif dataset_format == "tfrecord":
            data_set = ds.GeneratorDataset(generator_squad(data_file_path), shuffle=do_shuffle,
                                           column_names=["input_ids", "input_mask", "segment_ids", "unique_ids"])
        else:
            raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="unique_ids")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


def create_eval_dataset(batchsize=32, device_num=1, rank=0, data_dir=None, schema_dir=None,
                        dataset_format="mindrecord", num_samples=None):
    """create evaluation dataset"""
    data_files = []
    if os.path.isdir(data_dir):
        files = os.listdir(data_dir)
        for file_name in files:
            if (dataset_format == "tfrecord" and "tfrecord" in file_name) or \
                    (dataset_format == "mindrecord" and "mindrecord" in file_name and "mindrecord.db" not in file_name):
                data_files.append(os.path.join(data_dir, file_name))
    else:
        data_files.append(data_dir)
    if dataset_format == "mindrecord":
        if str(num_samples).lower() != "none":
            data_set = ds.MindDataset(data_files,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                      num_samples=num_samples)
        else:
            data_set = ds.MindDataset(data_files,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"])
    elif dataset_format == "tfrecord":
        data_set = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                      shard_equal_rows=True)
    else:
        raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")
    ori_dataset_size = data_set.get_dataset_size()
    print("origin eval size: ", ori_dataset_size)
    dtypes = data_set.output_types()
    shapes = data_set.output_shapes()
    output_batches = math.ceil(ori_dataset_size / device_num / batchsize)
    padded_num = output_batches * device_num * batchsize - ori_dataset_size
    print("padded num: ", padded_num)
    if padded_num > 0:
        item = {"input_ids": np.zeros(shapes[0], dtypes[0]),
                "input_mask": np.zeros(shapes[1], dtypes[1]),
                "segment_ids": np.zeros(shapes[2], dtypes[2]),
                "next_sentence_labels": np.zeros(shapes[3], dtypes[3]),
                "masked_lm_positions": np.zeros(shapes[4], dtypes[4]),
                "masked_lm_ids": np.zeros(shapes[5], dtypes[5]),
                "masked_lm_weights": np.zeros(shapes[6], dtypes[6])}
        padded_samples = [item for x in range(padded_num)]
        padded_ds = ds.PaddedDataset(padded_samples)
        eval_ds = data_set + padded_ds
        sampler = ds.DistributedSampler(num_shards=device_num, shard_id=rank, shuffle=False)
        eval_ds.use_sampler(sampler)
    else:
        if dataset_format == "mindrecord":
            if str(num_samples).lower() != "none":
                eval_ds = ds.MindDataset(data_files,
                                         columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                       "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                         num_shards=device_num, shard_id=rank, num_samples=num_samples)
            else:
                eval_ds = ds.MindDataset(data_files,
                                         columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                       "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                         num_shards=device_num, shard_id=rank)
        elif dataset_format == "tfrecord":
            eval_ds = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None,
                                         columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                       "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                         num_shards=device_num, shard_id=rank, shard_equal_rows=True)
        else:
            raise NotImplementedError("Only supported dataset_format for tfrecord or mindrecord.")

    type_cast_op = C.TypeCast(mstype.int32)
    eval_ds = eval_ds.map(input_columns="masked_lm_ids", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="masked_lm_positions", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="next_sentence_labels", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="segment_ids", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="input_mask", operations=type_cast_op)
    eval_ds = eval_ds.map(input_columns="input_ids", operations=type_cast_op)

    eval_ds = eval_ds.batch(batchsize, drop_remainder=True)
    print("eval data size: {}".format(eval_ds.get_dataset_size()))
    print("eval repeat count: {}".format(eval_ds.get_repeat_count()))
    return eval_ds
