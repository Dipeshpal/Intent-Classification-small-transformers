from datasets import load_dataset


def push_dataset():
    example = "Intent-Classification-Commands"
    print("Example of database name (you can name it anything): ", example)
    dataset_id = input("Enter your dataset name (huggingface): ")
    dataset = load_dataset('csv', data_files={'train': f'dataset_folder/train.csv', 'test': f'dataset_folder/test.csv'},
                           encoding="ISO-8859-1")
    dataset.push_to_hub(dataset_id)


if __name__ == '__main__':
    push_dataset()
