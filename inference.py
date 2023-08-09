import random, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from model import RNNModel
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pk
import csv

################################################################################
# Settings
################################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=999, help='manual random seed')
# parser.add_argument('--new_data', action='store_true')
# parser.add_argument('--is_test', action='store_true')
# parser.add_argument('--use_pretrain', action='store_true')


parser.add_argument('--input_dim', type=int, default=50)
parser.add_argument('--output_dim', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--rnn_type', type=str, default='gru')
parser.add_argument('--weight_decay', type=float, default=1e-6)


parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPU')

# parser.add_argument('--transcription-file', type=str)
# parser.add_argument('--output-file', type=str)

args = parser.parse_args()

# define a pytorch Dataset object
class SeqDataset(Dataset):
    def __init__(self, data, l_list):
        self.data = data
        self.l_list = l_list

    def __getitem__(self, index):
        return self.data[index], self.l_list[index]

    def __len__(self):
        return self.data.shape[0]

class Corpus(object):
    def __init__(self, rows, dictionary):
        texts = []

        for row in rows:
            text = row[3]
            texts.append(text.lower())

        self.pad_idx = 0
        self.dictionary = dictionary
        self.max_len = 0  # max length of all the sequences


        # tfidf feature for sequences
        self.feature = self.get_bow_feature([texts])[0]

        token_seqs = self.tokenize(texts)

        idx_seqs, self.l_list = self.get_seq_features(token_seqs)

        # l_list constains the last index, instead of length, which is 1 more
        self.max_len = max(self.l_list) + 1

        # indexed and padded sequences
        self.x = self.vectorize(idx_seqs)

        self.n_token = len(self.dictionary)

        # pretrained word embedding metrix
        # self.emb = self.get_emb(word2vec_file, emb_size=50)

        # embedding feature for sequences
        # self.emb = self.get_mean_emb(idx_seqs)

    def get_bow_feature(self, text_list):
        """
        Get TFIDF feature for a list of sequences
            input:
                text_list: a list of sequences
            output:
                data_list: a list of TFIDF features
        """

        corpus = []
        idxes = [0]
        for text in text_list:
            corpus.extend(text)
            idxes.append(idxes[-1] + len(text))

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(corpus)

        data_list = []
        for i in range(len(idxes) - 1):
            data_list.append(tfidf[idxes[i]:idxes[i + 1]])
        return data_list

    def tokenize(self, texts):
        """ Tokenize text file
        input:
            texts:  a list of sequences of tokens
        ouput:
            seqs: a list of lists of tokens
        """
        seqs = []
        for text in texts:
            seqs.append(word_tokenize(text))
        return seqs

    def get_seq_features(self, token_seqs):
        """
        Index sequences of tokens
        input:
            token_seqs: a list of lists of tokens
        output:
            idx_seqs: a list of lists of indexes
            l_list: a list of real length -1 of lists of tokens
        """

        idx_seqs = []
        l_list = []
        for seq in token_seqs:
            l = len(seq)
            l_list.append(l - 1)
            idx_list = []
            for token in seq:
                idx_list.append(self.dictionary.word2idx.get(token, 1))
            idx_seqs.append(idx_list)
        return idx_seqs, np.array(l_list, np.int32)

    def vectorize(self, idx_seqs):
        """ Convert sequences to numpy array and padding
        input:
            idx_sequences: a list of lists of indexes
        outputs:
            data: a numpy array containing all the lists padded with 0s
        """
        n_seq = len(idx_seqs)
        data = np.zeros((n_seq, self.max_len), dtype=np.int32)
        for i, word_ids in enumerate(idx_seqs):
            for j, word_id in enumerate(word_ids):
                data[i][j] = word_id
        return data

    def get_mean_emb(self, idx_seqs, emb_size=50):
        """Compute mean embedding for each sequence
        input:
            idx_seqs: a list of lists of indexes
        output:
            data: a numpy array of mean embeddings for sequences
        """

        n_seq = len(idx_seqs)
        data = np.zeros((n_seq, emb_size), dtype=np.float32)
        for i, word_ids in enumerate(idx_seqs):
            emb = np.zeros((1, emb_size), dtype=np.float32)
            for j, word_id in enumerate(word_ids):
                emb += self.emb[word_id]
            data[i] = emb / len(word_ids)
        return data


def infer(test_loader, net, device):
    net = net.to(device)

    print('Starting classification...')
    start_time = time.time()
    net.eval()
    predictions = []  # record predicted and true labels
    # start test
    with torch.no_grad():
        for i, (inputs, l_list) in enumerate(test_loader):
            inputs = inputs.to(device)
            l_list = l_list.to(device)

            outputs = net(inputs, l_list)

            labels_predict = torch.argmax(outputs, dim=1)

            predictions += labels_predict.cpu().data.numpy().tolist()

    test_time = time.time() - start_time
    print('Testing time: %.3f' % test_time)
    print('Finished classification.')

    return predictions


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)

    #######################################################################
    ######                     Prepare DATA                         #######
    #######################################################################


    start_time = time.time()
    datapath = args.transcription_file
    print('Loading data from {}'.format(datapath))
    with open(datapath, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]

    dictionary_path = 'dictionary.pkl'
    with open(dictionary_path, 'rb') as f:
        dictionary = pk.load(f)

    corpus = Corpus(rows, dictionary)
    print('Data time: {:.3f}'.format(time.time() - start_time))

    #######################################################################
    ######                        RNN Model                         #######
    #######################################################################

    # define model and output file name
    model_filename = "lr_0.001_id_50_od_128_ne_20_gru_nl_1_pretrain_True.pt"

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.n_gpu > 0) else "cpu")

    # prepare dataset
    data = SeqDataset(torch.LongTensor(corpus.x), torch.LongTensor(corpus.l_list))

    workers = 1
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    #######################################################################
    ######                           Model                           ######
    #######################################################################

    # model initialization
    net = RNNModel(corpus.n_token, args.input_dim, args.output_dim,
                   args.n_class, args.n_layer, args.rnn_type, device).to(device)

    checkpoint = torch.load(model_filename)
    net.load_state_dict(checkpoint['net_dict'])

    # testing process
    predictions = infer(loader, net, device)

    # record results
    assert len(predictions) == len(rows)

    index_to_label = {0: 'G', 1: 'A', 2: 'O'}

    outputpath = args.output_file
    print('Save result file to {}'.format(outputpath))
    with open(outputpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Start time', 'End time', 'Speaker', 'Transcription', 'Predicted Label'])

        for row, prediction in zip(rows, predictions):
            transcription = row[3]
            label = index_to_label[prediction]
            if transcription[0] == '[' and transcription[-1] == ']':
                label = 'O'
            writer.writerow(row + [label])

if __name__ == '__main__':
    main()