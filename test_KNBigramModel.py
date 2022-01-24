from unittest import TestCase
import KNBigramModel as kn

train_sam = kn.read_and_preprocess_corpus("Data/sam_kney.txt")

# Replace OOV words, and get the training vocab and unknown words
vocab, unknown_words = kn.replace_oov(train_sam)

sam_bigram_counts = kn.get_bigram_counts(train_sam)

class Test(TestCase):
    def test_see_bigram_counts(self):
        print(sam_bigram_counts)

    def test_get_words_in_columns(self):
        kn.build_model(sam_bigram_counts)

    def test_kneser_neys_like_sam(self):

        # bigram count
        bigram_count = sam_bigram_counts.loc["like", "sam"]

        # count w1:
        count_w1 = sam_bigram_counts.sum(axis=1)["like"]

        # number of non-zero bigram types
        non_zeroes = 0
        for column in sam_bigram_counts.columns:
            non_zeroes += sum(sam_bigram_counts[column] != 0)

        # word types that end with word2
        end_w2 = sum(sam_bigram_counts["sam"] != 0)

        # word types starting with word1
        # TODO

        self.assertTrue(0.35 == kn.kneser_neys(0.75, bigram_count, count_w1, non_zeroes, end_w2, 3))