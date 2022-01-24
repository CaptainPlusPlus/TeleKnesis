from unittest import TestCase
import KNBigramModel as kn

train_sam = kn.read_and_preprocess_corpus("Data/sam_kney.txt")

# Replace OOV words, and get the training vocab and unknown words
vocab, unknown_words = kn.replace_oov(train_sam)

class Test(TestCase):
    def test_see_bigram_counts(self):
        print(kn.get_bigram_counts(train_sam))
