import pandas as pd
import os


def get_intent_dict():
    file_path = "data_generator_from_various_sources/train.csv"

    df = pd.read_csv(file_path)

    # convert to dictionary intent: [text, text, text]
    intent_dict_temp = {}
    for index, row in df.iterrows():
        if row['label'] not in intent_dict_temp:
            intent_dict_temp[row['label']] = []
        intent_dict_temp[row['label']].append(row['text'])

    # pprint.pprint(intent_dict)
    print("TOTAL INTENTS: ", len(intent_dict_temp))
    # pprint.pprint(intent_dict_temp.keys())

    intent_dict = intent_dict_temp
    return intent_dict
