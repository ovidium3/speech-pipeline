"""
LINE 240, 241, 243: change to name of your file, or full pathname if that doesn't work

Audio should be in .wav format

The first time you will need to download "vader_lexicon" from nltk
nltk.download('vader_lexicon')
"""

import pandas as pd
import docx
import numpy as np
import glob
import opensmile
import os
import random, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from model import RNNModel
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pk
from textblob import TextBlob
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import moviepy.editor
import csv
from rpunct import RestorePuncts

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
            texts.append(row.lower())

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

def convert(length):
    hours = length // 3600
    stringhours = "0"+str(hours)
    length %= 3600
    minutes = length // 60
    if minutes < 10:
        stringminutes = "0"+str(minutes)
    else:
        stringminutes = str(minutes)
    length %= 60
    if length < 10:
        stringseconds = "0"+str(length)
    else:
        stringseconds = str(length)
    result = stringhours+":"+stringminutes+":"+stringseconds
    return result

def main():
    #option 1: the source file is in the format of m4a
    #manually convert to wav by instructions in "Readme.docx"

    #option 2: the source file is in the format of mp4, comment out the two lines if not
    clip = moviepy.editor.VideoFileClip("11_3.mp4")
    clip.audio.write_audiofile("11_3.wav")

    video = '11_3'  # change to name of word file WITHOUT .docx
    audio_ext = '.wav'
    transcript = docx.Document(video + '.docx')

    rpunct = RestorePuncts()

    lines = [p.text for p in transcript.paragraphs]

    while lines[0] != "Transcript":
        lines.pop(0)

    d = {'start': [], 'end':[], 'speaker': [], 'fulltext': []}

    last_speaker = 0
    i = 1
    while i < len(lines) - 1:
        first = lines[i]
        text = lines[i + 1]
        start = first[0:8]
        speaker = first[17:]
        if speaker == last_speaker:
            line = d['fulltext'][-1]
            while speaker == last_speaker:
                line = line + ' ' + text
                i += 2
                if i >= len(lines) - 1:
                    break
                first = lines[i]
                text = lines[i + 1]
                start = first[0:8]
                speaker = first[17:]
            d['fulltext'][-1] = rpunct.punctuate(line.translate(str.maketrans('', '', '!?,.')).lower(), lang='en')
        else:
            last_speaker = speaker
            d['start'].append(start)
            d['speaker'].append(speaker)
            d['fulltext'].append(text)
            i += 2

    d['end'] = d['start'].copy()
    d['end'].pop(0)
    audio = moviepy.editor.AudioFileClip(video + audio_ext)
    duration = int(audio.duration)
    d['end'].append(convert(duration))
    c2 = pd.DataFrame(data=d)

    #GAO
    random.seed(args.seed)
    np.random.seed(args.seed)

    #######################################################################
    ######                     Prepare DATA                         #######
    #######################################################################

    rows = c2['fulltext']

    dictionary_path = 'dictionary.pkl'
    with open(dictionary_path, 'rb') as f:
        dictionary = pk.load(f)

    corpus = Corpus(rows, dictionary)

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

    pdict = {'GAO':[]}
    for row, prediction in zip(rows, predictions):
        label = index_to_label[prediction]
        pdict['GAO'].append(prediction)

    pdf = pd.DataFrame(data=pdict)
    c2 = pd.concat([c2.reset_index(drop=True), pdf.reset_index(drop=True)], axis = 1)

    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    def getCharCount(text):
        return len(text)

    def getWordCount(text):
        return (len(text.split(' ')))

    sia = SentimentIntensityAnalyzer()

    def getnltkcompound(text):
        return (sia.polarity_scores(text)['compound'])

    def getnltkpos(text):
        return (sia.polarity_scores(text)['pos'])

    def getnltkneg(text):
        return (sia.polarity_scores(text)['neg'])

    def getnltkneu(text):
        return (sia.polarity_scores(text)['neu'])

    # Here, insert the file directory of the combined transcript you wish to get sentiment analysis from. You'll use the transcript you got from the "Merge Transcripts" R code

    # These lines are where we get the sentiment analysis for each line.
    c2['subjectivity'] = c2['fulltext'].apply(getSubjectivity)
    c2['polarity'] = c2['fulltext'].apply(getPolarity)
    c2['analysis'] = c2['polarity'].apply(getAnalysis)
    c2['nltkcompound'] = c2['fulltext'].apply(getnltkcompound)
    c2['nltkpos'] = c2['fulltext'].apply(getnltkpos)
    c2['nltkneg'] = c2['fulltext'].apply(getnltkneg)
    c2['nltkneu'] = c2['fulltext'].apply(getnltkneu)

    # This will save the file as .xlsx file. Pick the directory you want to save the file at, and name it "(team name)s.xlsx", to signifiy that that team has had its sentiment analysis completed..

    # writing times
    starts = c2['start']
    ends = c2['end']
    newstarts = [int(time[0:2]) * 3600 + int(time[3:5]) * 60 + int(time[6:]) for time in starts]
    newends = [int(time[0:2]) * 3600 + int(time[3:5]) * 60 + int(time[6:]) for time in ends]

    with open(video + '.txt', 'w') as f:
        for i in range(len(newstarts)):
            f.write(str(newstarts[i]) + '-' + str(newends[i]))
            f.write('\n')

    # splitting audio
    os.makedirs(video)
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    # Replace the filename below.
    required_video_file = video + audio_ext
    with open(video + '.txt') as f:
        times = f.readlines()
    times = [x.strip() for x in times]
    for i in range(len(times)):
        time = times[i]
        starttime = int(time.split("-")[0])
        endtime = int(time.split("-")[1])
        if endtime == starttime:
            endtime += 1
        ffmpeg_extract_subclip(required_video_file, starttime, endtime,
                               targetname=video + '/' + str(i) + audio_ext)

    # extracting
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    alldfs = []
    for i in range(len(times)):
        file = video + '/' + str(i) + audio_ext
        y = smile.process_file(file)
        alldfs.append(y)

    combined_csv = pd.concat(alldfs)

    final = pd.concat([c2.reset_index(drop=True), combined_csv.reset_index(drop=True)], axis=1)

    final.to_excel(video + '.xlsx')
    # Once you have this file, upload it to the appropriate _results folder in the One Drive.

#Not all of these packages are necessary, you will likely need to install textblob and NLTK
#To install textblob, follow the instructions on this website: https://anaconda.org/conda-forge/textblob or https://textblob.readthedocs.io/en/dev/install.html
#To install nltk, follow the instructions on this website: https://anaconda.org/anaconda/nltk or https://www.nltk.org/install.html
#The first time you run this code it will ask you to run a line to install "vader." Do this once ever, and your code should work.

#Import these packages

import nltk

#These lines define functions we'll use later. Run


if __name__ == '__main__':
    main()

