from unittest import TestCase

import Add1BigramModel
import Add1BigramModel as add1


class Test(TestCase):
    def test_read_and_preprocess_corpus_ja_sas_train(self):
        print(add1.read_and_preprocess_corpus("Data/ja-sas-train.txt"))
        self.assertTrue(True)

    def test_get_bigram_counts_show_output_sassy_train(self):
        training_data = add1.read_and_preprocess_corpus("Data/ja-sas-train.txt")
        print(add1.get_bigram_counts(training_data))
        self.assertTrue(True)

    def test_get_bigram_counts_show_output_sammy_train(self):
        training_data = add1.read_and_preprocess_corpus("Data/sam_i_am.txt")
        print(add1.get_bigram_counts(training_data))
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
        print(sentences)
        # print(sentences.count('sam'))
        self.assertTrue(True)

    def test_get_build_model_50_output(self):
        training_data = add1.read_and_preprocess_corpus("Data/50.txt")
        sentence = add1.generate_sentence(add1.build_model(add1.get_bigram_counts(training_data)), "cuck")
        print(sentence)
        self.assertTrue(True)

    def test_get_build_model_50_9_output(self):
        training_data = add1.read_and_preprocess_corpus("Data/50_9.txt")
        super_model = add1.build_model(add1.get_bigram_counts(training_data))
        for i in range(10):
            print(add1.generate_sentence(super_model, "cuck"))
        self.assertTrue(True)
