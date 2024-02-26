"""
Insert filename on line 11. File must be a wav file, as mp4 threw index error for some reason.
User_gservice_acc should remain constant, private_key_json differs based on user.
Speaker count on lines 24 + 25 can be estimated for greater accuracy, not necessary.
Sample rate hertz may need to be changed if getting error. Need to check the file if so.
//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 gpt_api_key
"""

from google.cloud import speech_v1p1beta1 as speech
import os
import opensmile
import pandas as pd

user_gservice_acc = "ovidiu-main@test-809-t5.iam.gserviceaccount.com"
private_key_json = "/Downloads/test-809-t5-2dfaa2d76624.json"

os.environ[user_gservice_acc] = private_key_json

client = speech.SpeechClient()

audio_file = "HRLR809_T5_2_sub_10"    # insert filename here
gcs_uri = "gs://test-809-t5-bucket/" + audio_file + ".wav"

audio = speech.RecognitionAudio(uri = gcs_uri)

diarization_config = speech.SpeakerDiarizationConfig(
    enable_speaker_diarization=True,
    min_speaker_count=2,    # change speakers here
    max_speaker_count=10,   # change speakers here
)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=32000, # change sample rate hertz here
    language_code="en-US",
    diarization_config=diarization_config,
)

operation = client.long_running_recognize(config = config, audio = audio)

print("Waiting for operation to complete...")
response = operation.result(timeout = 900) # arbitrary timeout value

# The transcript within each result is separate and sequential per result.
# However, the words list within an alternative includes all the words
# from all the results thus far. Thus, to get all the words with speaker
# tags, you only have to take the words list from the last result:
result = response.results[-1]

words_info = result.alternatives[0].words

# Printing out the individual words with speaker tags and timestamps:
for word_info in words_info:
    print("word: '{}', speaker_tag: {}, start_time: {}, end_time: {}".format(word_info.word, word_info.speaker_tag, word_info.start_time, word_info.end_time))

smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

alldfs = []
for word_info in words_info:
    y = smile.process_file(gcs_uri)
    alldfs.append(y)

combined_csv = pd.concat(alldfs)

final = pd.concat([c2.reset_index(drop=True), combined_csv.reset_index(drop=True)], axis=1)
# fifgure out what c2 is

final.to_excel(audio_file + '.xlsx')