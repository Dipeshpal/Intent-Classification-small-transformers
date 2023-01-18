import shutup
import pandas as pd
import tqdm
import json
from JARVISAI_INTENTS import data as JARVISAI_INTENTS
import plotly.express as px

shutup.please()


def get_hwu64_data():
    # get data from HWU64 dataset
    label_file_name = 'HWU64/train/label'
    label_file = open(label_file_name, 'r')
    label_list = label_file.readlines()
    label_file.close()

    text_file_name = 'HWU64/train/seq.in'
    text_file = open(text_file_name, 'r')
    text_list = text_file.readlines()
    text_file.close()

    for i in tqdm.tqdm(range(len(label_list))):
        label_list[i] = label_list[i].strip()
        text_list[i] = text_list[i].strip()

    df = pd.DataFrame({'text': text_list, 'label': label_list})
    df.to_csv('HWU64/train.csv', index=False)
    print("HWU64 data generated")


def get_jarvisai_data():
    intent_dict = JARVISAI_INTENTS.intent_dict
    df = pd.DataFrame(columns=['text', 'label'])
    for intent, keywords in intent_dict.items():
        for keyword in keywords:
            df = df.append({'text': keyword, 'label': intent.replace(" ", "_")}, ignore_index=True)
    df.to_csv('JARVISAI_INTENTS/train.csv', index=False)
    print("JARVISAI data generated")


def get_chatbot_data():
    file_path = 'chatbot/intent.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    data_intents = data['intents']
    df = pd.DataFrame(columns=['text', 'label'])
    for intent in data_intents:
        label = intent['intent'].replace(" ", "_").lower()
        text = intent['text']
        for t in text:
            df = df.append({'text': t, 'label': label}, ignore_index=True)
    df.to_csv('chatbot/train.csv', index=False)
    print("chatbot data generated")


def merge_data():
    df1 = pd.read_csv('HWU64/train.csv')
    df2 = pd.read_csv('JARVISAI_INTENTS/train.csv')
    df3 = pd.read_csv('chatbot/train.csv')
    df = pd.concat([df1, df2, df3], ignore_index=True)
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    # more shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('train.csv', index=False)
    print("data merged")


def plot_data():
    df = pd.read_csv('train.csv')
    fig = px.histogram(df, x="label")
    fig.show()


if __name__ == '__main__':
    get_hwu64_data()
    get_jarvisai_data()
    get_chatbot_data()
    merge_data()
    plot_data()
