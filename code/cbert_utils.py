import os
import csv
import logging
import random

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

"""initialize logger"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, label=None):
        """Constructs a InputExample/

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels):
        self.init_ids = init_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of 'InputExample's for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of 'InputExample's for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class AugProcessor(DataProcessor):
    """Processor for dataset to be augmented."""

    def get_train_examples(self, data_dir):
        """See base calss."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_labels(self, name):
        """add your dataset here"""
        if name in ['stsa.binary', 'mpqa', 'rt-polarity', 'subj']:
            return ["0", "1"]
        elif name in ['stsa.fine']:
            return ["0", "1", "2", "3", "4"]
        elif name in ['TREC']:
            return ["0", "1", "2", "3", "4", "5"]
    
    def _create_examples(self, lines, set_type):
        """Create examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of 'InputBatch's."""
    
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    
    features = []
    for (ex_index, example) in enumerate(examples):
        # The convention in BERT is:
        # tokens:   [CLS] is this jack ##son ##ville ? [SEP]
        # type_ids: 0     0  0    0    0     0       0 0    
        tokens_a = tokenizer._tokenize(example.text_a)
        tokens_label = label_map[example.label]
        tokens, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels = \
            extract_features(tokens_a, tokens_label, max_seq_length, tokenizer)
        
        """convert label to label_id"""
        label_id = label_map[example.label]

        """consturct features"""
        features.append(
            InputFeature(
                init_ids=init_ids,        
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                masked_lm_labels=masked_lm_labels))

        """print examples"""
        if ex_index < 5:
            logger.info("[cbert] *** Example ***")
            logger.info("[cbert] guid: %s" % (example.guid))
            logger.info("[cbert] tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("[cbert] init_ids: %s" % " ".join([str(x) for x in init_ids]))
            logger.info("[cbert] input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("[cbert] input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("[cbert] segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("[cbert] masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))
    return features

def construct_train_dataloader(train_examples, label_list, max_seq_length, train_batch_size, num_train_epochs, tokenizer, device):
    """construct dataloader for training data"""
    
    num_train_steps = None
    global_step = 0
    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer)
    num_train_steps = int(len(train_features) / train_batch_size * num_train_epochs)
    
    all_init_ids = torch.tensor([f.init_ids for f in train_features], dtype=torch.long, device=device)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long, device=device)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long, device=device)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long, device=device)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in train_features], dtype=torch.long, device=device)
    
    tensor_dataset = TensorDataset(all_init_ids, all_input_ids, all_input_mask, 
        all_segment_ids, all_masked_lm_labels)
    train_sampler = RandomSampler(tensor_dataset)
    train_dataloader = DataLoader(tensor_dataset, sampler=train_sampler, batch_size=train_batch_size)
    return train_features, num_train_steps, train_dataloader

def rev_wordpiece(str):
    """wordpiece function used in cbert"""
    
    #print(str)
    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if str[i] == '[PAD]':
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0]=='#' and str[i][1]=='#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return " ".join(str[1:-1])


def extract_features(tokens_a, tokens_label, max_seq_length, tokenizer):
    """extract features from tokens"""

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]
    
    tokens = []
    segment_ids = []
    tokens.append('[CLS]')
    segment_ids.append(tokens_label)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(tokens_label)
    tokens.append('[SEP]')
    segment_ids.append(tokens_label)

    ## construct init_ids for each example
    init_ids = convert_tokens_to_ids(tokens, tokenizer)

    ## construct input_ids for each example, we replace the word_id using 
    ## the ids of masked words (mask words based on original sentence)
    masked_lm_probs = 0.15
    max_predictions_per_seq = 20
    rng = random.Random(12345)
    original_masked_lm_labels = [-1] * max_seq_length
    (output_tokens, masked_lm_positions, 
    masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_probs, original_masked_lm_labels, max_predictions_per_seq, rng, tokenizer)
    input_ids = convert_tokens_to_ids(output_tokens, tokenizer)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        init_ids.append(0)
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    assert len(init_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return tokens, init_ids, input_ids, input_mask, segment_ids, masked_lm_labels

def convert_tokens_to_ids(tokens, tokenizer):
    """Converts tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        token_id = tokenizer._convert_token_to_id(token)
        ids.append(token_id)
    return ids

def create_masked_lm_predictions(tokens, masked_lm_probs, masked_lm_labels, 
                                 max_predictions_per_seq, rng, tokenizer):
    """Creates the predictions for the masked LM objective."""

    #vocab_words = list(tokenizer.vocab.keys())
    
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)
    
    rng.shuffle(cand_indexes)
    len_cand = len(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq, 
                         max(1, int(round(len(tokens) * masked_lm_probs))))
    
    masked_lm_positions = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lm_positions) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        ## 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            ## 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            ## 10% of the time, replace with random word
            else:
                masked_token = tokens[cand_indexes[rng.randint(0, len_cand - 1)]]
                
        masked_lm_labels[index] = convert_tokens_to_ids([tokens[index]], tokenizer)[0]
        output_tokens[index] = masked_token
        masked_lm_positions.append(index)
    return output_tokens, masked_lm_positions, masked_lm_labels