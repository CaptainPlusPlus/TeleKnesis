"""
Course:         Statistical Methods for NLP I: Parsing
Project:        Bigram Models
Author(s):      <Your first and last name(s) and matriculation number(s)>
Description:    <very short description of what the code does (e.g. Bigram model
                with add-1 smoothing)>
Honor Code:     I/We pledge that this code is my/our own work, and that
                no part of this work was copied from, or shared with, others.
"""

import argparse
import random
import math
import pandas as pd
import numpy as np


BOS_MARKER = "<s>"
EOS_MARKER = "</s>"
UNKNOWN = "<UNK>"


# ToDo:
#  Generate a sentence with probabilities according
#  to the given bigram model
#  The returned sentence is a list of words ['<s>', 'w1', ..., 'wn', '</s>']
#  Replace UNKNOWN with a randomly selected word from unknown_words.
def generate_sentence(prob_df, unknown_words):
    return []


# ToDo:
#  Return the log probability of sent,
#  where sent is a list of words [<s>, 'w1', ..., 'wn', </s>].
def get_sent_logprob(log_df, sent):
    return -np.inf


# ToDo:
#  Calculate the perplexity of the model with the given test sentences
def get_perplexity(log_df, test_sentences):
    return np.inf


# Return a DataFrame representing the unigram counts for the
# tokenized training data.
def get_unigram_counts(training_data):
    # counts is a dict: {key=word, value=count}
    counts = {}

    for sent in training_data:
        for tok in sent:
            counts[tok] = counts.get(tok, 0) + 1

    # Convert dict to dataframe and return the dataframe
    count_df = pd.DataFrame(counts, index=[0])
    return count_df


# Replace words that only appear once in the training data with UNKNOWN.
# This is done by by generating unigram counts, then combining into one
# column (with the special name UNKNOWN), all columns with count==1.
# Return the vocabulary of the training data and a list of unknown words.
def replace_oov(training_data):

    count_df = get_unigram_counts(training_data)

    # Words that appear only once are considered UNKNOWN
    # Get the column headers (words) with count == 1
    row0 = count_df.iloc[0]
    unknown_words = row0.index.values[row0 == 1]
    num_unknown_words = len(unknown_words)

    # drop unknown words columns
    count_df.drop(unknown_words, axis=1, inplace=True)

    # Create an UNKNOWN column (with sum of unknown words)
    count_df[UNKNOWN] = num_unknown_words

    # Vocabulary is a list of the column names
    vocab = list(count_df.columns)

    return vocab, unknown_words


# Return a dataframe of bigram counts, where
# rows represent w1 and columns represent w2.
# If a vocabulary is given, replace words that are not in the vocabulary with UNKNOWN.
# If padding=True, insert BOS and EOS sentence markers.
def get_bigram_counts(training_data, vocab=None, padding=True):

    # counts is a nested dict: {key=w_i, value = {key=w_j, value=count}}
    counts = {}

    for sentence in training_data:
        if vocab:
            sent = [w if w in vocab else UNKNOWN for w in sentence]
        else:
            sent = sentence

        if padding:
            sent.insert(0, BOS_MARKER)
            sent.append(EOS_MARKER)

        # iterate sent bigrams: (w1,w2), (w2,w3), (w3,w4)...
        for (w_i, w_j) in zip(sent, sent[1:]):
            counts[w_i] = counts.setdefault(w_i, {w_j: 0})
            counts[w_i][w_j] = counts[w_i].setdefault(w_j, 0) + 1

    count_df = pd.DataFrame(counts).T
    count_df.replace(np.nan, 0, inplace=True)
    return count_df


# Convert all dataframe cells to log probabilities using base 10
def df_to_log(prob_df):
    return np.log10(prob_df)


# Read corpus file line-by-line and perform preprocessing
# Returns a list of tokenized sentences
# If vocab is provided, oov words are replaced with UNKNOWN
# If padding=True, add BOS and EOS sentence markers
def read_and_preprocess_corpus(corpus_file, vocab=None, padding=False):
    with open(corpus_file, "r", encoding="utf-8") as corpus:
        sentences = corpus.readlines()

    tok_sentences = []
    for sent in sentences:
        tok_sent = sent.lower()
        tok_sent = tok_sent.split()
        if vocab:
            tok_sent = [tok if tok in vocab else UNKNOWN for tok in tok_sent]
        if padding:
            tok_sent.insert(0, BOS_MARKER)
            tok_sent.append(EOS_MARKER)
        tok_sentences.append(tok_sent)

    return tok_sentences


# ToDo:
#  Build the model using the given bigram counts dataframe,
#  and using the smoothing method described in the project requirements.
def build_model(bigram_counts, d=.75):
    # ToDo (add-1):
    #  The parameter d is only needed for Kneser-Ney smoothing, and can be
    #  ignored for add-1 smoothing.
    #  You do not need to build a dictionary for add-1 smoothing. The
    #  built-in dataframe operations are fast and efficient in this case.
    #
    # ToDo (Kneser-Ney):
    #  Build the model using a nested dictionary (see get_bigram_counts for
    #  an example of how to build a nested dictionary), then convert the
    #  dictionary to a dataframe and return the dataframe.
    #  It is necessary to use a dictionary because creating a dataframe and
    #  updating/adding values in it is MUCH TOO SLOW.
    #  Avoid making the same calculations over and over. Some values can be
    #  reused (for example, continuation probabilities only need to be
    #  calculated once).

    return pd.DataFrame()


# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_file", help="corpus file")
    parser.add_argument("test_file", help="test file")
    return parser.parse_args()


# ToDo:
#  Train and test a bigram model.
#  The submitted models should not contain any print statements that are
#  not in this starter code. Please remove extra print statements that
#  you may have added during development before submission.
def main(args):

    # Read the training data
    train = read_and_preprocess_corpus(args.corpus_file)

    # Replace OOV words, and get the training vocab and unknown words
    vocab, unknown_words = replace_oov(train)

    # ToDo: calculate and print the OOV rate of the training corpus
    oov_rate = 0
    print(f'OOV rate: {oov_rate}')

    # Get the bigram counts, using the fixed vocabulary
    bigram_counts = get_bigram_counts(train, vocab=vocab)

    # ToDo: build the model
    bigram_model = build_model(bigram_counts)

    # convert probabilities to log probabilities
    log_df = df_to_log(bigram_model)

    # Generate and print 10 sentences using the model
    for i in range(0, 10):
        # use probabilities to generate sentences
        sent = generate_sentence(bigram_model, unknown_words)
        sent = ' '.join(sent)
        print(f'{sent}')

    # Read the test corpus, using the fixed vocabulary, and adding sentence markers
    test_sentences = read_and_preprocess_corpus(args.test_file, vocab=vocab, padding=True)

    # Print the sentence probabilities for the first 5 test sentences
    for test_sent in test_sentences[:5]:
        # use log probabilities to get prob of a sentence
        logprob = get_sent_logprob(log_df, test_sent)
        print(f'P({test_sent}): {math.pow(10, logprob)}')

    # Calculate the perplexity of the test sentences corpus
    # Use log probabilities to calculate perplexity
    perplexity = get_perplexity(log_df, test_sentences)
    print(f'\nperplexity of test_sentences: {perplexity}')


if __name__ == '__main__':
    main(parse_args())