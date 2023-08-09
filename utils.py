class Dictionary(object):
    """
    A dictionary class for index tokens in the sequences
    """
    def __init__(self):
        self.word2idx = {'PAD_IDX':0, 'UNK_IDX':1}
        self.idx2word = {0:'PAD_IDX', 1:'UNK_IDX'}
        self.pad = 0
        self.idx = 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[self.idx] = word
            self.word2idx[word] = self.idx # pad
            self.idx += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word.keys())