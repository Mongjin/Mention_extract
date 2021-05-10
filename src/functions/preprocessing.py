import torch
import os
import config
from tqdm import tqdm
from torch.utils.data import TensorDataset

open_vocab_path = os.path.join(config.data_dir, "open_vocab.txt")
close_vocab_path = os.path.join(config.data_dir, "close_vocab.txt")

open_label_map = {"[PAD]": 0}
close_label_map = {"[PAD]": 0}

with open(open_vocab_path, 'r', encoding='utf-8') as vocab:
    for label in vocab:
        label = label.strip()
        open_label_map[label] = len(open_label_map)

with open(close_vocab_path, 'r', encoding='utf-8') as vocab:
    for label in vocab:
        label = label.strip()
        close_label_map[label] = len(close_label_map)


# 학습 or 평가 데이터를 읽어 리스트에 저장
def read_data(file_path, mode):
    with open(file_path, "r", encoding="utf8") as inFile:
        lines = inFile.readlines()

    datas = []
    for index, line in enumerate(tqdm(lines, desc="read_data")):
        # 입력 문장을 \t으로 분리
        pieces = line.strip().split("\t")

        if mode == "train" or mode == "test":
            # 데이터의 형태가 올바른지 체크
            assert len(pieces) == 3
            if len(pieces[0].split(" ")) == len(pieces[1].split(" ")) == len(pieces[2].split(" ")):
                continue
            sentence, open_label, close_label = pieces[0], [open_label_map[label] for label in pieces[1].split(" ")], [close_label_map[label] for label in pieces[2].split(" ")]
            datas.append((sentence, open_label, close_label))
        # elif mode == "analyze":
        #     sentence, senti_label = pieces[0], int(pieces[1])
        #     datas.append((sentence, senti_label))
    return datas


def convert_data2dataset(datas, tokenizer, max_length, open_labels, close_labels, mode):
    total_input_ids, total_attention_mask, total_token_type_ids, total_open_labels, total_close_labels, total_open_seq, total_close_seq, total_word_seq = [], [], [], [], [], [], [], []

    if mode == "analyze":
        total_open_labels = None
        total_close_labels = None
    for index, data in enumerate(tqdm(datas, desc="convert_data2dataset")):
        sentence = ""
        open_label = []
        close_label = []
        if mode == "train" or mode == "test":
            sentence, open_label, close_label = data
        # elif mode == "analyze":
        #     sentence, open_label = data

        tokens = []
        open_ids = []
        close_ids = []
        for word, open_tag, close_tag in zip(sentence.split(" "), open_label, close_label):
            word_tokens = tokenizer.tokenize(word.lower())
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
            open_ids.extend([open_tag] + [0] * (len(word_tokens)-1))
            close_ids.extend([close_tag] + [0] * (len(word_tokens)-1))

        tokens = ["[CLS]"] + tokens
        tokens = tokens[:max_length-1]
        open_ids = [open_label_map["[PAD]"]] + open_ids
        open_ids = open_ids[:max_length-1]
        close_ids = [close_label_map["[PAD]"]] + close_ids
        close_ids = close_ids[:max_length-1]
        tokens.append("[SEP]")
        open_ids.append(open_label_map["[PAD]"])
        close_ids.append(close_label_map["[PAD]"])
        input_ids = [tokenizer._convert_token_to_id(token) for token in tokens]
        assert len(input_ids) <= max_length

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding = [0] * (max_length - len(input_ids))

        total_word_seq.append(len(input_ids))

        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)

        total_open_seq.append([i for i in range(open_labels)])
        total_close_seq.append([i for i in range(close_labels)])

        if mode == "train" or mode == "test":
            open_ids += padding
            close_ids += padding
            total_open_labels.append(open_ids)
            total_close_labels.append(close_ids)

        if index < 2:
            print("*** Example ***")
            print("sequence : {}".format(sentence))
            print("input_ids: {}".format(" ".join([str(x) for x in total_input_ids[-1]])))
            print("attention_mask: {}".format(" ".join([str(x) for x in total_attention_mask[-1]])))
            print("token_type_ids: {}".format(" ".join([str(x) for x in total_token_type_ids[-1]])))
            print("open_seq: {}".format(total_open_seq[-1]))
            print("close_seq: {}".format(total_close_seq[-1]))
            print("word_seq: {}".format(total_word_seq[-1]))
            print()
            if mode == "train" or mode == "test":
                print("open_label: {}".format(total_open_labels[-1]))
                print("close_label: {}".format(total_close_labels[-1]))

    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_open_seq = torch.tensor(total_open_seq, dtype=torch.long)
    total_close_seq = torch.tensor(total_close_seq, dtype=torch.long)
    total_word_seq = torch.tensor(total_word_seq, dtype=torch.long)
    if mode == "train" or mode == "test":
        total_open_labels = torch.tensor(total_open_labels, dtype=torch.long)
        total_close_labels = torch.tensor(total_close_labels, dtype=torch.long)
        dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_open_labels,
                            total_close_labels, total_open_seq, total_close_seq, total_word_seq)
    # elif mode == "analyze":
    #     dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_senti_labels,
    #                             total_senti_seq, total_score_seq, total_word_seq)

    return dataset
