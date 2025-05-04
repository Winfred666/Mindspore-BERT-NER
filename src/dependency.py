import argparse
import os
import codecs
import pickle
import numpy as np


import ast

from pprint import pformat
import yaml

import math
import collections

import copy


# WARNING!!! if called by do_one_test, use src. else called by run_ner, delete src.
import src.tokenization as tokenization

__all__ = ['NerProcessor', 'write_tokens', 'convert_single_example', 'filed_based_convert_examples_to_features']




class InputFeatures():
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()

class InputExample():
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


def write_tokens(tokens, output_dir, mode):
    """
    write token result to output txt
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode,vocab_file=None):
    """
    convert example to id single by single
    """
    # convert the label into index.
    label_map = {}
    for (i, label) in enumerate(label_list, 0):
        label_map[label] = i

    textlist = example.text.split(' ')
    labellist = example.label.split(' ') # this may contain some labels that we do not want.

    # Warning: In this we will

    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)

        label_1 = labellist[i] # extract one label output example, will not contain label we do not want.
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                print("Generate sub-token: ", word)
                labels.append("X") # if extract multiple token, use X for sub-token (seldom happen)
            
    # if the length of one line(seq) exceed the max-seq-len, just cut it off(what a waste)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []

    # Do not add [CLS] when there is already "start" and "stop" token for CLR layer.
    
    # ntokens.append("[CLS]")  # add [CLS] begin of token
    # segment_ids.append(0)
    # label_ids.append(label_map["[CLS]"])

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    
    # ntokens.append("[SEP]")  # add [SEP] end of token
    # segment_ids.append(0)
    # label_ids.append(label_map["[SEP]"])

    if(vocab_file == None):
        vocab_file = args_opt.vocab_file
    input_ids = tokenization.convert_tokens_to_ids(vocab_file, ntokens)  # convert ntokens to ID format
    input_mask = [1] * len(input_ids)
    # padding for unrelated (WARNING: 0 must be O tag for no real meaning.)
    padding_labels_id =  label_map['O']
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # WARNING: For meaningless padding token, the label_id should be index of 'O'
        label_ids.append(padding_labels_id)
        
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )

    write_tokens(ntokens, output_dir, mode)
    return feature



# use BMES representation, however can use other to do it.
def convert_labels_to_index(label_list):
    """
    Convert label_list to indices for NER task.
    """
    label2id = collections.OrderedDict()
    label2id["O"] = 0
    # prefix = ["S_", "B_", "M_", "E_"]
    index = 0
    for label in label_list:
        if(label == "O"):
            continue
        # for pre in prefix:
        index += 1
        sub_label = label
        label2id[sub_label] = index
    return label2id









def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="pretrain_base_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
            # print(cfg_helper)
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def extra_operations(cfg):
    """
    Do extra work on config

    Args:
        config: Object after instantiation of class 'Config'.
    """
    def create_filter_fun(keywords):
        return lambda x: not (True in [key in x.name.lower() for key in keywords])

    if cfg.description == 'run_pretrain':
        cfg.AdamWeightDecay.decay_filter = create_filter_fun(cfg.AdamWeightDecay.decay_filter)
        cfg.Lamb.decay_filter = create_filter_fun(cfg.Lamb.decay_filter)
        cfg.base_net_cfg.dtype = parse_dtype(cfg.base_net_cfg.dtype)
        cfg.base_net_cfg.compute_type = parse_dtype(cfg.base_net_cfg.compute_type)
        cfg.nezha_net_cfg.dtype = parse_dtype(cfg.nezha_net_cfg.dtype)
        cfg.nezha_net_cfg.compute_type = parse_dtype(cfg.nezha_net_cfg.compute_type)
        cfg.large_net_cfg.dtype = parse_dtype(cfg.large_net_cfg.dtype)
        cfg.large_net_cfg.compute_type = parse_dtype(cfg.large_net_cfg.compute_type)
        cfg.large_boost_net_cfg.dtype = parse_dtype(cfg.large_boost_net_cfg.dtype)
        cfg.large_boost_net_cfg.compute_type = parse_dtype(cfg.large_boost_net_cfg.compute_type)
        if cfg.bert_network == 'base':
            cfg.batch_size = cfg.base_batch_size
            _bert_net_cfg = cfg.base_net_cfg
        elif cfg.bert_network == 'nezha':
            cfg.batch_size = cfg.nezha_batch_size
            _bert_net_cfg = cfg.nezha_net_cfg
        elif cfg.bert_network == 'large':
            cfg.batch_size = cfg.large_batch_size
            _bert_net_cfg = cfg.large_net_cfg
        elif cfg.bert_network == 'large_boost':
            cfg.batch_size = cfg.large_boost_batch_size
            _bert_net_cfg = cfg.large_boost_net_cfg
        else:
            pass
        cfg.bert_net_cfg = BertConfig(**_bert_net_cfg.__dict__)
    elif cfg.description == 'run_ner':
        cfg.optimizer_cfg.AdamWeightDecay.decay_filter = \
            create_filter_fun(cfg.optimizer_cfg.AdamWeightDecay.decay_filter)
        cfg.optimizer_cfg.Lamb.decay_filter = create_filter_fun(cfg.optimizer_cfg.Lamb.decay_filter)
        cfg.bert_net_cfg.dtype = mstype.float32
        cfg.bert_net_cfg.compute_type = mstype.float16
        cfg.bert_net_cfg = BertConfig(**cfg.bert_net_cfg.__dict__)

    elif cfg.description == 'run_squad':
        cfg.optimizer_cfg.AdamWeightDecay.decay_filter = \
            create_filter_fun(cfg.optimizer_cfg.AdamWeightDecay.decay_filter)
        cfg.optimizer_cfg.Lamb.decay_filter = create_filter_fun(cfg.optimizer_cfg.Lamb.decay_filter)
        cfg.bert_net_cfg.dtype = mstype.float32
        cfg.bert_net_cfg.compute_type = mstype.float16
        cfg.bert_net_cfg = BertConfig(**cfg.bert_net_cfg.__dict__)

    elif cfg.description == 'run_classifier':
        cfg.optimizer_cfg.AdamWeightDecay.decay_filter = \
            create_filter_fun(cfg.optimizer_cfg.AdamWeightDecay.decay_filter)
        cfg.optimizer_cfg.Lamb.decay_filter = create_filter_fun(cfg.optimizer_cfg.Lamb.decay_filter)
        cfg.bert_net_cfg.dtype = mstype.float32
        cfg.bert_net_cfg.compute_type = mstype.float16
        cfg.bert_net_cfg = BertConfig(**cfg.bert_net_cfg.__dict__)
    else:
        pass


def parse_dtype(dtype):
    if dtype not in ["mstype.float32", "mstype.float16"]:
        raise ValueError("Not supported dtype")

    if dtype == "mstype.float32":
        return mstype.float32
    if dtype == "mstype.float16":
        return mstype.float16
    return None

def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    def get_abs_path(path_relative):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, path_relative)
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", type=get_abs_path, default="../pretrain_config.yaml",
                        help="Config file path")
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    config_obj = Config(final_config)
    extra_operations(config_obj)
    return config_obj


# config = get_config()
# bert_net_cfg = config.bert_net_cfg
# if config.description in ('run_classifier', 'run_ner', 'run_squad'):
#     optimizer_cfg = config.optimizer_cfg




# class BertNER(nn.Cell):
#     """
#     Train interface for sequence labeling finetuning task.
#     """

#     def __init__(self, config, batch_size, is_training, num_labels=11, use_crf=False, with_lstm=False,
#                  tag_to_index=None, dropout_prob=0.0, use_one_hot_embeddings=False):
#         super(BertNER, self).__init__()
#         # this model can add LSTM at the end.
#         self.bert = BertNERModel(config, is_training, num_labels, use_crf, with_lstm, dropout_prob,
#                                  use_one_hot_embeddings)
#         # add a CRF layer.
#         if use_crf:
#             if not tag_to_index:
#                 raise Exception("The dict for tag-index mapping should be provided for CRF.")
#             from src.CRF import CRF
#             self.loss = CRF(tag_to_index, batch_size, config.seq_length, is_training)
#         else:
#             self.loss = CrossEntropyCalculation(is_training)
#         self.num_labels = num_labels
#         self.use_crf = use_crf

#     def construct(self, input_ids, input_mask, token_type_id, label_ids,real_seq_length):
#         # WARNING: when use lstm, tell the model what is the actual length of each sequence.
#         logits = self.bert(input_ids, input_mask, token_type_id,real_seq_length)
        
#         if self.use_crf:
#             loss = self.loss(logits, label_ids)
#         else:
#             loss = self.loss(logits, label_ids, self.num_labels)
#         return loss








# class BertConfig:
#     """
#     Configuration for `BertModel`.

#     Args:
#         seq_length (int): Length of input sequence. Default: 128.
#         vocab_size (int): The shape of each embedding vector. Default: 32000.
#         hidden_size (int): Size of the bert encoder layers. Default: 768.
#         num_hidden_layers (int): Number of hidden layers in the BertTransformer encoder
#                            cell. Default: 12.
#         num_attention_heads (int): Number of attention heads in the BertTransformer
#                              encoder cell. Default: 12.
#         intermediate_size (int): Size of intermediate layer in the BertTransformer
#                            encoder cell. Default: 3072.
#         hidden_act (str): Activation function used in the BertTransformer encoder
#                     cell. Default: "gelu".
#         hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
#         attention_probs_dropout_prob (float): The dropout probability for
#                                       BertAttention. Default: 0.1.
#         max_position_embeddings (int): Maximum length of sequences used in this
#                                  model. Default: 512.
#         type_vocab_size (int): Size of token type vocab. Default: 16.
#         initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
#         use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
#         dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
#         compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
#     """
#     def __init__(self,
#                  seq_length=128,
#                  vocab_size=32000,
#                  hidden_size=768,
#                  num_hidden_layers=12,
#                  num_attention_heads=12,
#                  intermediate_size=3072,
#                  hidden_act="gelu",
#                  hidden_dropout_prob=0.1,
#                  attention_probs_dropout_prob=0.1,
#                  max_position_embeddings=512,
#                  type_vocab_size=16,
#                  initializer_range=0.02,
#                  use_relative_positions=False,
#                  dtype=mstype.float32,
#                  compute_type=mstype.float32):
#         self.seq_length = seq_length
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.hidden_act = hidden_act
#         self.intermediate_size = intermediate_size
#         self.hidden_dropout_prob = hidden_dropout_prob
#         self.attention_probs_dropout_prob = attention_probs_dropout_prob
#         self.max_position_embeddings = max_position_embeddings
#         self.type_vocab_size = type_vocab_size
#         self.initializer_range = initializer_range
#         self.use_relative_positions = use_relative_positions
#         self.dtype = dtype
#         self.compute_type = compute_type
