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
