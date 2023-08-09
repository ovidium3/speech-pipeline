import pandas as pd
import numpy as np

data = pd.read_excel('MCL26.xlsx')  # change file name
speakers = pd.read_excel('MCL27speakers.xlsx')  # change file name

speaker_dict = dict(zip(speakers['speaker#'],speakers['roster i.d.']))

new_speakers = []
for speaker in data['speaker']:
    if np.isnan(speaker):
        new_speakers.append(np.nan)
    else:
        new_speakers.append(speaker_dict[speaker])

data.insert(3, 'roster', new_speakers)
data = data.drop(data.columns[[0]], axis=1)

data.to_excel('MCL26withspeaker.xlsx')  # change file name



