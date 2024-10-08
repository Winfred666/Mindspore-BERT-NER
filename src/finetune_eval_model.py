# Copyright 2020 Huawei Technologies Co., Ltd
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

'''
Bert finetune and evaluation model script.
'''
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore import context

from .bert_model import BertModel

import mindspore as ms


# Build QuantiDCE model(finetuned, student version, only use KD loss)
class BertQuantiDCEModel(nn.Cell):
    """
    This is mindspore version of task Dialogue Coherence Evaluation
    : https://arxiv.org/abs/2106.00507
    """
    def __init__(self,config, is_training, dropout_prob=0.0, use_one_hot_embeddings=False,):
        super(BertQuantiDCEModel, self).__init__()
        
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        
        self.bert = BertModel(config, is_training,use_one_hot_embeddings)
        self.cast = P.Cast()
        
        self.dropout = nn.Dropout(1 - dropout_prob)
        
        # A kind of weight_init method for the comming layer like mlp
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        # self.num_labels = 1 # no label but score will be the output.
        # this is also the bert_hidden_size, 128
        self.hidden_size = config.hidden_size
        mlp_hidden_size_1 = int(self.hidden_size / 2) # gradually thrinking the output to 32 
        mlp_hidden_size_2 = int(mlp_hidden_size_1 / 2)

        self.mlp = nn.SequentialCell(
            nn.Dense(self.hidden_size, mlp_hidden_size_1,weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type),
            nn.LeakyReLU(alpha=0.1),
            nn.Dense(mlp_hidden_size_1, mlp_hidden_size_2,weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type),
            nn.LeakyReLU(alpha=0.1),
            nn.Dense(mlp_hidden_size_2, 1, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type), # only one score output
            nn.Sigmoid())
    
    def construct(self, input_ids, token_type_id, input_mask):
        #  the pooled_output usually the embedding of the [CLS] token (the first token)
        _, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        # Cast type to float32 that GPU support.
        cls = self.cast(pooled_output, self.dtype)
        # drop out some node weight for training
        cls = self.dropout(cls)
        score = self.mlp(cls)
        score = self.cast(score, self.dtype)
        return score


# "CLS" sometimes means the first output embedding, that is , the vector of [CLS]
# However here just for Classification, use a dense layer(Classification layer) to convert logits space to label space.
class BertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2)(Our target, relative or irrelative.), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        # num_labels is much smaller than hidden_size.
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.assessment_method = assessment_method

    def construct(self, input_ids, input_mask, token_type_id):
        # first go throught basic bert.
        _, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        # Cast type to float32 that GPU support.
        cls = self.cast(pooled_output, self.dtype)
        # drop out some node weight for training
        cls = self.dropout(cls)
        # go through one dense_1 to convert hidden_size to label_size.
        logits = self.dense_1(cls)
        # cast again for GPU support.
        logits = self.cast(logits, self.dtype)
        # finally a softmax function make logits into probability distribution.
        if self.assessment_method != "spearman_correlation":
            logits = self.log_softmax(logits)
        return logits


class BertSquadModel(nn.Cell):
    '''
    This class is responsible for SQuAD
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSquadModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense1 = nn.Dense(config.hidden_size, num_labels, weight_init=self.weight_init,
                               has_bias=True).to_float(config.compute_type)
        self.num_labels = num_labels
        self.dtype = config.dtype
        self.log_softmax = P.LogSoftmax(axis=1)
        self.is_training = is_training
        self.gpu_target = context.get_context("device_target") == "GPU"
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.shape = (-1, config.hidden_size)
        self.origin_shape = (-1, config.seq_length, self.num_labels)
        self.transpose_shape = (-1, self.num_labels, config.seq_length)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        sequence = self.reshape(sequence_output, self.shape)
        logits = self.dense1(sequence)
        logits = self.cast(logits, self.dtype)
        logits = self.reshape(logits, self.origin_shape)
        if self.gpu_target:
            logits = self.transpose(logits, (0, 2, 1))
            logits = self.log_softmax(self.reshape(logits, (-1, self.transpose_shape[-1])))
            logits = self.transpose(self.reshape(logits, self.transpose_shape), (0, 2, 1))
        else:
            logits = self.log_softmax(logits)
        return logits


class BertNERModel(nn.Cell):
    """
    This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
    The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=11, use_crf=False, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertNERModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        
        # last classification layer
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
                
        if with_lstm:
            self.lstm_hidden_size = config.hidden_size // 2
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_hidden_size, has_bias=True,
             batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.use_crf = use_crf
        self.with_lstm = with_lstm
        self.origin_shape = (-1, config.seq_length, self.num_labels)

        # the biggest length of one sequence, 
        # however the padding token is not recommanded to use, 
        # so need to explicit specify the actual length of sequence, 
        # so that Bi-LSTM will not gather state from empty list when doing back to forth
        
        # self.seq_length = ms.Tensor(self.seq_length,ms.int64)

    def construct(self, input_ids, input_mask, token_type_id, real_seq_length):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)

        if self.with_lstm:
            batch_size = input_ids.shape[0]
            data_type = self.dtype
            hidden_size = self.lstm_hidden_size
            # 2 is for bidirectional.
            # provide h0 and c0 at both end, which is by default zero.
            h0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            c0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            
            # here the seq:[batch_size , max_seq_length , embedding_hidden_dim]
            seq, _ = self.lstm(seq, (h0,c0) , seq_length = real_seq_length)
        
        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        if self.use_crf:
            return_value = self.reshape(logits, self.origin_shape)
        else:
            return_value = self.log_softmax(logits)
        return return_value
