import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import sys
import argparse
import os
import logging
from tqdm import tqdm


def init_logging_path(log_path, file_name):
    '''
    build log file
    :param log_path: log path
    :param file_name: log file
    :return:
    '''
    dir_log = os.path.join(log_path, f"{file_name}/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
            os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
            os.utime(dir_log, None)
    return dir_log


def find_token_id(input_id, tokenizer):
    token_pos_start_id = set([tokenizer.encode('[', add_special_tokens=False)[0], tokenizer.encode(' [', add_special_tokens=False)[0]])
    token_pos_end_id = set([tokenizer.encode(']', add_special_tokens=False)[0], tokenizer.encode(' ]', add_special_tokens=False)[0]])

    token_ids = []
    for i, input_i in enumerate(input_id):
        input_i = int(input_i)
        if i == len(input_id) - 1:  # the last token
            continue
        if input_i in [tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]:
            continue
        if input_i in token_pos_start_id:
            token_ids.append(i + 1)
            # logger.info("first word",token_ids)
        elif input_i in token_pos_end_id:
            token_ids.append(i)
    try:
        assert len(token_ids) == 2
    except AssertionError as e:
        print('Warning, token id alter is not length 2')
        print(input_id)
        print(tokenizer.convert_ids_to_tokens(input_id))
        print(token_pos_start_id)
        print(token_pos_end_id)
        print(token_ids)
        sys.exit(1)

    try:
        assert token_ids[1] != token_ids[0]
    except AssertionError as e:
        print('token marker star == end')
        print(input_id)
        print(token_ids)
        sys.exit(1)
    token_ids[1] = token_ids[1] - 1
    token_ids[0] = token_ids[0] - 1
    return token_ids


def delete_tokenmark_input(input_ids,tokenizer):
    input_id_new=[]
    del_num=0
    token_pos_start_id=[tokenizer.encode('[', add_special_tokens=False)[0],tokenizer.encode(' [',add_special_tokens=False)[0]]
    token_pos_end_id=[tokenizer.encode(']', add_special_tokens=False)[0],tokenizer.encode(' ]',add_special_tokens=False)[0]]
    token_pos_start_end_id=set(token_pos_start_id+token_pos_end_id)
    for i,input_i in enumerate(input_ids):
        if input_i not in token_pos_start_end_id:
            input_id_new.append(input_i)
        else:
            del_num+=1
    input_id_new+=del_num*[tokenizer.pad_token_id]
    return input_id_new


def delete_tokenmarker_am(input_ids,tokenizer):
    am_new=[]
    for i in input_ids:
        if i==tokenizer.pad_token_id:
            am_new.append(0)
        else:
            am_new.append(1)
    return am_new


def run(args):
    logging.info(str(args))
    logging.info("load word file")
    words = []
    with open(args.w_file, 'r', encoding='utf-8') as f:
        for line in f:
            words.append(line.strip().lower())

    logging.info("total word number is " + str(len(words)))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, output_hidden_states=True).eval()

    use_gpu = torch.cuda.is_available()

    cnt = 0
    type_emb = []
    for w in words:
        if cnt > 0 and cnt % 100 == 0:
            logging.info(str(cnt) + " of " + str(len(words)) + " processed.")
        if w + '.txt' not in os.listdir(args.sent_path):
            continue
        texts = []
        with open(os.path.join(args.sent_path, w + ".txt"), 'r', encoding='utf-8') as f:
            for line in f:
                l, _, r = line.strip().replace('[', '').replace(' [', ' ').replace(']', '').replace('] ', ' ').partition(w)
                # if left part of word is longer than max_seq_len, remove first half of tokens
                tokens = tokenizer.tokenize(l)
                if len(tokens) >= args.max_seq_len - 4:  # 4 -> cls, sep, [, ]
                    l = ' '.join(tokens[-((args.max_seq_len - 4) // 2):]) + " "
                texts.append(l + "[ " + w + " ]" + r)
        # print(w)
        # texts = texts[:5]
        emb = torch.tensor([])
        for i in tqdm(range(0, len(texts), args.batch_size)):
            toks = tokenizer.batch_encode_plus(texts[i:i + args.batch_size], max_length=args.max_seq_len, truncation=True, padding="max_length")
            target_token_ids = torch.tensor([find_token_id(tok, tokenizer) for tok in toks['input_ids']], dtype=torch.long)
            all_input_ids = torch.tensor([delete_tokenmark_input(tok, tokenizer) for tok in toks['input_ids']],
                                        dtype=torch.long)
            all_attention_mask = torch.tensor([delete_tokenmarker_am(input_ids, tokenizer) for input_ids in all_input_ids],
                                            dtype=torch.long)

            if use_gpu:
                all_input_ids = all_input_ids.to("cuda")
                all_attention_mask = all_attention_mask.to("cuda")
                model.to("cuda")

            inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_mask}
            with torch.no_grad():
                outputs_ = model(**inputs, output_hidden_states=True)
            hidden_states = outputs_.hidden_states
            average_layer_batch = sum(hidden_states[args.start_layer:args.end_layer]) / (args.end_layer - args.start_layer)

            for num in range(average_layer_batch.size()[0]):
                embeds_per_sent = average_layer_batch[num]
                token_ids_per_sent = target_token_ids[num]

                embed_token = torch.mean(embeds_per_sent[int(token_ids_per_sent[0]):int(token_ids_per_sent[1])], dim=0,
                                        keepdim=True)
                assert not torch.isnan(embed_token).any()
                if num == 0:
                    output = embed_token
                else:
                    output = torch.cat((output, embed_token), 0)
            emb = torch.cat((emb, output.detach().cpu()))

        torch.save(emb, os.path.join(args.out_path, w + ".pt"))
        type_emb.append(torch.mean(emb, dim=0).detach().cpu().numpy())
        cnt += 1
        
    np.save('mirrowwic.npy', np.array(type_emb))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sent_path", help="sent_path", default="mcrae_sents")
    parser.add_argument("-w_file", help="file name of word list", default="abstract.txt")
    parser.add_argument("-max_seq_len", help="max sequence length", type=int, default=128)
    parser.add_argument("-start_layer", help="start layer", type=int, default=9)
    parser.add_argument("-end_layer", help="end layer", type=int, default=13)
    parser.add_argument("-out_path", help="output path of mv", default="mv_mirrorwic")
    parser.add_argument("-model", help="model path or model name", default="cambridgeltl/mirrorwic-bert-base-uncased")  # cambridgeltl/mirrorwic-bert-base-uncased
    parser.add_argument("-batch_size", help="batch size", default=32, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    log_file_path = init_logging_path("log", "exmv_mirrorwic")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    run(args)

