import random
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import os
from util import *


def dataset_generator():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists('dataset_folder'):
        os.mkdir('dataset_folder')

    BALANCE_DATA = input("Do you want to balance the data? (y/n): ")
    if BALANCE_DATA == 'y':
        BALANCE_DATA = True
    else:
        BALANCE_DATA = False

    base_path = 'dataset_folder'
    folder_path = f'{base_path}/data_BALANCE_DATA-{BALANCE_DATA}'
    file_name = 'dataset.csv'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


    print('Creating intent dataframe...')
    col1 = []
    col2 = []
    intent_dict = get_intent_dict()
    for key in intent_dict.keys():
        for value in intent_dict[key]:
            col1.append(value)
            col2.append(key)

    # list to dataframe
    df = pd.DataFrame({'text': col1, 'intent': col2})

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    print("Initial Dataframe Shape: ", df.shape)

    print('label encoding...')
    le = preprocessing.LabelEncoder()
    df['labels'] = le.fit_transform(df['intent'])

    if BALANCE_DATA:
        print("Frequency of each label, close window to continue")
        df['labels'].value_counts().plot(kind='bar')
        plt.show()

        print('Balancing and Up-sampling the dataset...')
        value_counts = df.intent.value_counts()
        print(value_counts)
        max_value = value_counts.max()
        print(max_value)

        # max_value = max_value * 4

        di = {}

        for i in df.intent.unique():
            if di.get(i, None) is None:
                di[i] = intent_dict[i]

            while len(di[i]) < max_value:
                di[i].append(random.choice(intent_dict[i]))

        # creating final dataframe
        print('Creating final dataframe')
        col1 = []
        col2 = []

        for key in intent_dict.keys():
            for value in intent_dict[key]:
                col1.append(value)
                col2.append(key)

    # list to dataframe
    df = pd.DataFrame({'text': col1, 'intent': col2})

    # label encoding
    print('label encoding...')
    le = preprocessing.LabelEncoder()
    df['labels'] = le.fit_transform(df['intent'])
    df.to_csv(f'{folder_path}/{file_name}', index=False)

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    print(df.shape)
    df.to_csv(f'{folder_path}/{file_name}', index=False)

    # Train Test Split
    print('Train Test Split Dataset...')
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(f'{folder_path}/train.csv', index=False)
    test.to_csv(f'{folder_path}/test.csv', index=False)
    print('Train Test Split Dataset Completed')

    print('Train Dataset Shape: ', train.shape)
    print('Test Dataset Shape: ', test.shape)

    print("Generating Labels...")
    df = pd.read_csv(f'{folder_path}/{file_name}')

    dict_labels = {}

    for index, row in df.iterrows():
        if row['labels'] not in dict_labels:
            dict_labels[row['labels']] = row['intent']

    # to json
    with open(f'{folder_path}/labels.json', 'w') as f:
        json.dump(dict_labels, f)

    # load json file
    with open(f'{folder_path}/labels.json', 'r') as f:
        dict_labels = json.load(f)

    print('Done')

    print("Value Counts of each label")
    # value counts
    print(df.intent.value_counts())
    print("Frequency of each label, close window to continue")
    df['labels'].value_counts().plot(kind='bar')
    plt.show()
