import math
from unittest import TestCase

import Add1BigramModel
import Add1BigramModel as add1


class Test(TestCase):
    def test_read_and_preprocess_corpus_ja_sas_train(self):
      # print(add1.read_and_preprocess_corpus("Data/ja-sas-train.txt"))
        self.assertTrue(True)

    def test_get_bigram_counts_show_output_sassy_train(self):
        training_data = add1.read_and_preprocess_corpus("Data/ja-sas-train.txt")
      # print(add1.get_bigram_counts(training_data))
        self.assertTrue(True)

    def test_get_bigram_counts_show_output_sammy_train(self):
        training_data = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
      # print(add1.get_bigram_counts(training_data))
        self.assertTrue(True)

    def test_get_build_model_sam_i_am_output(self):
        training_data = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
        add1.build_model(add1.get_bigram_counts(training_data))
        self.assertTrue(True)

    def test_get_build_model_sam_i_am_output(self):
        training_data = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
        sentences = []
        super_model = add1.build_model(add1.get_bigram_counts(training_data))
        for i in range(100): sentences += [add1.generate_sentence(super_model, "cuck")]
      # print(sentences)
        # print(sentences.count('sam'))
        self.assertTrue(True)

    def test_get_build_model_50_output(self):
        training_data = add1.read_and_preprocess_corpus("Data/50.txt")
        sentence = add1.generate_sentence(add1.build_model(add1.get_bigram_counts(training_data)), "cuck")
      # print(sentence)
        self.assertTrue(True)

    def test_get_build_model_50_9_output(self):
        training_data = add1.read_and_preprocess_corpus("Data/50_9.txt")
        super_model = add1.build_model(add1.get_bigram_counts(training_data))
        #for i in range(10):
            # print(add1.generate_sentence(super_model, "cuck"))
        self.assertTrue(True)

    def test_see_if_unk_is_in_probdf(self):
        # Read the training data
        train = add1.read_and_preprocess_corpus("Data/sam_i_am_test")

        # Replace OOV words, and get the training vocab and unknown words
        vocab, unknown_words = add1.replace_oov(train)

        # Get the bigram counts, using the fixed vocabulary
        bigram_counts = add1.get_bigram_counts(train, vocab=vocab)

        # ToDo: build the model
        bigram_model = add1.build_model(bigram_counts)

        # convert probabilities to log probabilities
        log_df = add1.df_to_log(bigram_model)

        # Generate and print 10 sentences using the model
        for i in range(0, 10):
            # use probabilities to generate sentences
            sent = add1.generate_sentence(bigram_model, unknown_words)
            sent = ' '.join(sent)
            # print(f'{sent}')

        # Read the test corpus, using the fixed vocabulary, and adding sentence markers
        test_sentences = add1.read_and_preprocess_corpus("Data/sam_i_am_test", vocab=vocab, padding=True)

    def test_get_sent_prob_log_first_try(self):
        train = add1.read_and_preprocess_corpus("Data/ja-sas-train.txt")
        vocab, unknown_words = add1.replace_oov(train)
        bigram_counts = add1.get_bigram_counts(train, vocab)
        model = add1.build_model(bigram_counts)
        log_df = add1.df_to_log(model)
        # print(log_df)
        our_log_prob = add1.get_sent_logprob(log_df, ["<s>", "i", "cannot", "believe", "this", "work", "</s>"])
        expected = log_df.loc["<s>", "i"] + log_df.loc["i", "cannot"] + log_df.loc["cannot", "believe"] \
                   + log_df.loc["believe", "this"] + log_df.loc["this", "<UNK>"] + log_df.loc["<UNK>", "</s>"]
        # print(our_log_prob, expected)
        self.assertTrue(our_log_prob == expected)

    def test_get_sent_prob_log_sec_try(self):
        train = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
        vocab, unknown_words = add1.replace_oov(train)
        bigram_counts = add1.get_bigram_counts(train, vocab)
        model = add1.build_model(bigram_counts)
        log_df = add1.df_to_log(model)

        test_sentences = add1.read_and_preprocess_corpus("Data/sam_i_am_test", vocab=vocab, padding=True)

        # Print the sentence probabilities for the first 5 test sentences
        for test_sent in test_sentences[:1]:
            # use log probabilities to get prob of a sentence
            logprob = add1.get_sent_logprob(log_df, test_sent)
          # print(f'P({test_sent}): {math.pow(10, logprob)}')

    def test_get_perplexity_value(self):
        train = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
        vocab, unknown_words = add1.replace_oov(train)
        bigram_counts = add1.get_bigram_counts(train, vocab)
        model = add1.build_model(bigram_counts)
        log_df = add1.df_to_log(model)
        test_sentences = add1.read_and_preprocess_corpus("Data/sam_i_am_test", vocab=vocab, padding=True)
        self.assertIsNotNone(add1.get_perplexity(log_df, test_sentences))

    def test_get_perplexity_value(self):
        train = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
        vocab, unknown_words = add1.replace_oov(train)
        bigram_counts = add1.get_bigram_counts(train, vocab)
        model = add1.build_model(bigram_counts)
        log_df = add1.df_to_log(model)
        print(model)
        test_sentences = add1.read_and_preprocess_corpus("Data/sam_i_am_test", vocab=vocab, padding=True)
        self.assertIsNotNone(add1.get_perplexity(log_df, test_sentences))

    def test_get_perplexity_value_stam(self):
        train = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
        vocab, unknown_words = add1.replace_oov(train)
        bigram_counts = add1.get_bigram_counts(train, vocab)
        model = add1.build_model(bigram_counts)
        log_df = add1.df_to_log(model)
        print(model)
        test_sentences = add1.read_and_preprocess_corpus("Data/sam_i_am.txt", vocab=vocab, padding=True)
        self.assertIsNotNone(add1.get_perplexity(log_df, test_sentences))


    def test_get_perplexity_value_multi_sentence_corpus(self):
        train = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
        vocab, unknown_words = add1.replace_oov(train)
        bigram_counts = add1.get_bigram_counts(train, vocab)
        model = add1.build_model(bigram_counts)
        log_df = add1.df_to_log(model)
        test_sentences = add1.read_and_preprocess_corpus("Data/sam_i_am_test_multiple_sams", vocab=vocab, padding=True)
        self.assertIsNotNone(add1.get_perplexity(log_df, test_sentences))

    def test_get_perplexity_value_pap(self):
        train = add1.read_and_preprocess_corpus("Data/ja-sas-train.txt")
        vocab, unknown_words = add1.replace_oov(train)
        bigram_counts = add1.get_bigram_counts(train, vocab)
        model = add1.build_model(bigram_counts)
        log_df = add1.df_to_log(model)
        test_sentences = add1.read_and_preprocess_corpus("Data/ja-pap-test.txt", vocab=vocab, padding=True)
        self.assertIsNotNone(add1.get_perplexity(log_df, test_sentences))
