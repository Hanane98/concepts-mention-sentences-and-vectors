## Extract sentences containing nouns in bert vocabulary from wikipedia text
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk import pos_tag, word_tokenize, sent_tokenize
import os
import logging
import sys

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
import getopt
import urllib.parse


def write_txt(output_file, contents):
    with open(output_file, 'a+', encoding='utf-8') as f:
        f.write('\n'.join(contents)+'\n')


def main(argv):
    max_sentence_length = 64
    corpusfile = 'wikipedia-sentences/sentences_nouns_30.txt'
    nounsfile = 'missing_1.txt'
    outputdir = 'sumo_sent'
    file_miss_word = 'missing_word.txt'

    try:
        opts, args = getopt.getopt(argv, "hl:n:c:o:f:", ["lfile=", "nfile=", "cfile=", "odir=", "fmissword="])
    except getopt.GetoptError:
        print('extract_senteces.py  -l <logfile> -n <nounfile> -c <corpusfile> -o <outputdir> -f <fmissword>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('extract_senteces.py  -l <logfile> -n <nounfile> -c <corpusfile> -o <outputdir> -f <fmissword>')
            sys.exit()
        elif opt in ("-n", "--nfile"):
            nounsfile = arg
        elif opt in ("-c", "--cfile"):
            corpusfile = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg
        elif opt in ("-f", "--fmiss"):
            file_miss_word = arg

    corpus_file_path = corpusfile
    nouns_file_path = nounsfile
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    logging.info('noun file is ' + nounsfile)
    logging.info('corpus file is ' + corpusfile)
    logging.info('output dire is ' + outputdir)

    # noun.txt: one noun per line
    nouns = []
    with open(nouns_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            nouns.append(line.strip())
    nouns = set(nouns)
    logging.info('number of words: ' + str(len(nouns)))

    words_have = set()
    with open(corpus_file_path, 'r', encoding='utf-8') as f:
        cnt = 0
        noun_sentences = {}
        for line in f:
            cnt += 1
            if cnt % 100000 == 0:
                logging.info(str(round(cnt / 1000000, 4)) + "M lines processed")
                for key in noun_sentences:
                    out_file = os.path.join(outputdir, urllib.parse.quote_plus(key) + '.txt')
                    write_txt(out_file, noun_sentences[key])
                    words_have.add(key)
                noun_sentences = {}
            sent = line.strip().lower().split('\t')[0]
            if len(sent.split(" ")) > max_sentence_length:
                continue

            for w in nouns:
                if len(w) == 1:
                    continue
                if sent.find(" " + w + " ") > 0:
                    for word, pos in pos_tag(word_tokenize(sent)):
                        if word == w and pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                            if w not in noun_sentences:
                                noun_sentences[w] = [sent]
                            elif len(noun_sentences[w]) < 500 and sent not in noun_sentences[w]:
                                noun_sentences[w].append(sent)

                else:
                    if '_' in w:  # e.g. hot_dog
                        n1 = w.replace("_", "")  # e.g. hotdog
                        n2 = w.replace("_", " ")  # e.g. hot dog

                        if sent.find(" " + n1 + " ") > 0:
                            sent = sent.replace(" " + n1 + " ", " " + w + " ")
                            if w not in noun_sentences:
                                noun_sentences[w] = [sent]
                            elif len(noun_sentences[w]) < 500 and sent not in noun_sentences[w]:
                                noun_sentences[w].append(sent)

                        elif sent.find(" " + n2 + " ") > 0:
                            sent = sent.replace(" " + n2 + " ", " " + w + " ")
                            if w not in noun_sentences:
                                noun_sentences[w] = [sent]
                            elif len(noun_sentences[w]) < 500 and sent not in noun_sentences[w]:
                                noun_sentences[w].append(sent)

        if cnt % 100000 > 0:
            for key in noun_sentences:
                out_file = os.path.join(outputdir, urllib.parse.quote_plus(key) + '.txt')
                write_txt(out_file, noun_sentences[key])
                words_have.add(key)

    with open(file_miss_word, 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(set(nouns)-set(words_have))))
    print("DONE GETTING CONTEXTS")


if __name__ == '__main__':
    main(sys.argv[1:])
