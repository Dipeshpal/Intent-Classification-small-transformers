import os
import pandas as pd
from transformers import pipeline
import torch

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# write a function to keep or remove the intent from the dataset if in li
li = ['play_music', 'play_games', 'recommendation_locations', 'news_query',
      'general_quirky', 'asking_weather', 'audio_volume_mute', 'music_query',
      'play_game', 'datetime_query', 'general_explain', 'weather_query',
      'courtesygreeting', 'volume_control', 'audio_volume_down',
      'email_sendemail', 'tell_me_news', 'general_joke',
      'greetingresponse', 'email_query', 'tell_me_about', 'tell_me_joke',
      'i_am_bored', 'send_email',
      'jokes', 'take_screenshot', 'play_on_youtube', 'what_can_you_do',
      'asking_time', 'covid_cases', 'asking_date', 'goodbye',
      'download_youtube_video', 'shutup', 'greeting', 'timequery',
      'click_photo', 'places_near_me', 'courtesygreetingresponse', 'open_website',
      'send_whatsapp_message', 'courtesygoodbye',
      'greet_and_general_check_in']

features = [
    'asking date', 'asking time', 'tell me joke', 'tell me news', 'asking weather',
    'tell me about', 'open website', 'play on youtube', 'send whatsapp message',
    'send email', 'greet and hello hi kind of things, general check in', 'goodbye',
    # 'conversation': chatbot.chatbot_general_purpose,
    'take screenshot', 'click photo',
    # 'check internet speed': internet_speed_test.speed_test,
    'download youtube video', 'covid cases', 'play games', 'places near me', 'i am bored',
    'volume control', 'what can you do'
]

map_dict = {
    'play_music': 'play on youtube',
    'play_games': 'play games',
    'recommendation_locations': 'places near me',
    'news_query': 'tell me news',
    'general_quirky': 'tell me about',
    'asking_weather': 'asking weather',
    'audio_volume_mute': 'volume control',
    'music_query': 'play on youtube',
    'play_game': 'play games',
    'datetime_query': 'asking time',
    'general_explain': 'tell me about',
    'weather_query': 'asking weather',
    'courtesygreeting': 'greet and hello hi kind of things, general check in',
    'volume_control': 'volume control',
    'audio_volume_down': 'volume control',
    'email_sendemail': 'send email',
    'tell_me_news': 'tell me news',
    'general_joke': 'tell me joke',
    'greetingresponse': 'greet and hello hi kind of things, general check in',
    'email_query': 'send email',
    'tell_me_about': 'tell me about',
    'tell_me_joke': 'tell me joke',
    'i_am_bored': 'i am bored',
    'send_email': 'send email',
    'jokes': 'tell me joke',
    'take_screenshot': 'take screenshot',
    'play_on_youtube': 'play on youtube',
    'what_can_you_do': 'what can you do',
    'asking_time': 'asking time',
    'covid_cases': 'covid cases',
    'asking_date': 'asking date',
    'goodbye': 'goodbye',
    'download_youtube_video': 'download youtube video',
    'shutup': 'goodbye',
    'greeting': 'greet and hello hi kind of things, general check in',
    'timequery': 'asking time',
    'click_photo': 'click photo',
    'places_near_me': 'places near me',
    'courtesygreetingresponse': 'greet and hello hi kind of things, general check in',
    'open_website': 'open website',
    'send_whatsapp_message': 'send whatsapp message',
    'courtesygoodbye': 'goodbye',
    'greet_and_general_check_in': 'greet and hello hi kind of things, general check in'
}


# check if map_dict keys are in li
for i in map_dict.keys():
    if i not in li:
        print("Keys: ", i)

# check if map_dict values are in features
for i in map_dict.values():
    if i not in features:
        print("Values: ", i)


def filter_dataset():
    csv_data = "dataset_folder/data_BALANCE_DATA-True/dataset.csv"

    df = pd.read_csv(csv_data)

    print(df.shape)
    print(len(df.intent.unique()))

    uniques = df.intent.unique()

    for i in uniques:
        if i not in li:
            df = df[df.intent != i]

    # save the new dataset
    df.to_csv("dataset_folder/data_BALANCE_DATA-True/dataset-filtered.csv", index=False)

    print(df.shape)
    print(len(df.intent.unique()))


def rename_columns():
    csv_data = "dataset_folder/data_BALANCE_DATA-True/dataset-filtered.csv"

    df = pd.read_csv(csv_data)
    df = df.rename(columns={'intent': 'intent_old'})

    df['intent'] = df['intent_old'].map(map_dict)

    # remove the old intent column
    df = df.drop(columns=['intent_old', 'labels'])

    # do a label encoding
    df['labels'] = df['intent'].astype('category')
    df['labels'] = df['labels'].cat.codes

    # drop the rows with labels = -1
    df = df[df.labels != -1]
    print(df.labels.unique())

    # save the new dataset
    df.to_csv("dataset_folder/data_BALANCE_DATA-True/dataset-filtered-renamed.csv", index=False)


#

if __name__ == "__main__":
    # filter_dataset()
    rename_columns()
