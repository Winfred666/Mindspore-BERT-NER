# Copyright 2022 Huawei Technologies Co., Ltd
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

import argparse
import os
import codecs
import pickle
import numpy as np
from mindspore.mindrecord import FileWriter

# WARNING!!! if called by do_one_test, use src. else called by run_ner, delete src.
import src.tokenization as tokenization

__all__ = ['NerProcessor', 'write_tokens', 'convert_single_example', 'filed_based_convert_examples_to_features']


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


class InputFeatures():
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids



class NerProcessor():
    def __init__(self, output_dir,labels=None):
        if isinstance(labels, list):
            self.labels = labels
        else:
            self.labels = []
        self.label_used = [False] * len(self.labels)
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "example.train")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "example.dev")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "example.test")), "test")

    def get_labels(self, labels=None):

        # This is where well go through
        if self.labels:
            # Do not include [CLS] because already use <START> and <STOP>
            # self.labels = self.labels.union(set(["X", "[CLS]", "[SEP]"]))
            print(self.labels)
            with open(os.path.join(self.output_dir, 'label_list.txt'), 'w') as rf:
                for label in self.labels:
                    rf.write(label + "\n")
        
        
        else:
            self.labels = ["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X",
                            "[CLS]", "[SEP]"]
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # tokenization
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            
            self.label_used = [False] * len(self.labels)

            # every line is just one token.
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])

                else: # if no contends(means the end of a sequence), and we have accumulate some token, then make a seq.
                    if not contends and words:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if l and w:
                                # WARNING: Do not change indicated self.labels.
                                # self.labels.add(l)
                                # here we going to trainslate labels to index, if there are un-interested token, make them to O
                                if l not in self.labels:
                                    l = 'O'
                                else :
                                    self.label_used[self.labels.index(l)] = True
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            # report the labels that no use in the file
            for i in range(len(self.label_used)):
                if( not self.label_used[i]):
                    print(f"Label {self.labels[i]} is not used!!!")
            return lines


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


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
    """
    convert examples to mindrecord format
    """
    # Add a column of real_seq_length, so that bi_lstm can be optimized !!
    schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "input_mask": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "label_ids": {"type": "int32", "shape": [-1]},
        "real_seq_length":{"type": "int32", "shape":[1]}, # this is only one integer, but for standard type encapsulate it as a list.
    }
    
    writer = FileWriter(output_file, overwrite=True)
    writer.add_schema(schema)
    total_written = 0

    for (ex_index, example) in enumerate(examples):
        all_data = []
        # this is the REAL entry for convert raw_dataset into mindrecord. also call by do_one_test.py
        
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)
        input_ids = np.array(feature.input_ids, dtype=np.int32)
        input_mask = np.array(feature.input_mask, dtype=np.int32)
        segment_ids = np.array(feature.segment_ids, dtype=np.int32)
        label_ids = np.array(feature.label_ids, dtype=np.int32)
        
        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_ids": label_ids,
                "real_seq_length":np.array([len(example.text.split()) + 2],dtype=np.int32)
        } # +2 is for <START> (no [CLR] [SEP]) <END>.
        # print(len(example.text.split()) + 2)

        all_data.append(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1
    writer.commit()
    # this is the final part.
    print("Total instances is: ", total_written, flush=True)

def main(args):
    # check output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # delete the repeated token while maintain the order !!
    processor = NerProcessor(args.output_dir, labels=list(dict.fromkeys(args.labels.split())))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    eval_examples = None

    if args.do_train and args.do_eval:
        train_examples = processor.get_train_examples(args.data_dir)

        print("***** Running training *****")
        print("  Num examples is: ", len(train_examples), flush=True)

        eval_examples = processor.get_dev_examples(args.data_dir)
        print("***** Running evaluation *****")
        print("  Num examples is: ", len(eval_examples), flush=True)

        test_examples = processor.get_test_examples(args.data_dir)
        print("***** Running test *****")
        print("  Num examples is: ", len(test_examples), flush=True)


    # get label list !!!
    label_list = processor.get_labels()

    # no matter how, write the file.
    train_file = os.path.join(args.output_dir, "train.mind_record")
    filed_based_convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, train_file, args.output_dir)

    eval_file = os.path.join(args.output_dir, "eval.mind_record")
    filed_based_convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, eval_file, args.output_dir)

    test_file = os.path.join(args.output_dir, "test.mind_record")
    filed_based_convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, test_file, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make dataset in mindrecord format.')
    parser.add_argument('--data_dir', default=".", type=str, help='')
    parser.add_argument('--max_seq_length', default=202, type=int, help='')
    parser.add_argument('--do_train', default=True, type=bool, help='')
    parser.add_argument('--do_eval', default=True, type=bool, help='')
    parser.add_argument('--do_lower_case', default=True, type=bool, help='')
    parser.add_argument('--vocab_file', default="./vocab.txt", type=str, help='')
    parser.add_argument('--output_dir', default="./outputs", type=str, help='')
    parser.add_argument('--labels', default="", type=str, help='')
    
    args_opt = parser.parse_args()
    main(args_opt)
