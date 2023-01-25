import os
import pandas as pd
from transformers import pipeline
import torch

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("CPU or GPU:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print(torch.cuda.current_device())
torch.cuda.set_device(1)
print(torch.cuda.current_device())


# write a function to keep or remove the intent from the dataset if in li
li = ['play_music', 'play_games', 'recommendation_locations', 'news_query', 'iot_coffee',
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


classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


# def util_rename_using_ai(sequence_to_classify):
#     res = classifier(sequence_to_classify, features)
#     return res['labels'][0]


def rename_columns():
    csv_data = "dataset_folder/data_BALANCE_DATA-True/dataset-filtered.csv"

    df = pd.read_csv(csv_data)
    df = df.rename(columns={'intent': 'intent_old'})

    for index, row in df.iterrows():
        sequence = row['intent_old']
        res = classifier(sequence, features)
        df.at[index, 'intent'] = res['labels'][0]
        print(f"{sequence} -> {res['labels'][0]}")

    df.to_csv("dataset_folder/data_BALANCE_DATA-True/dataset-filtered-renamed.csv", index=False)


#

if __name__ == "__main__":
    # filter_dataset()
    rename_columns()
