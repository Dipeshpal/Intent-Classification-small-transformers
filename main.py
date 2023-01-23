from data_generator_from_various_sources import script
from dataset_generator import dataset_generator


def make_data_from_sources():
    script.get_hwu64_data()
    script.get_jarvisai_data()
    script.get_chatbot_data()
    script.merge_data()
    script.plot_data()


def make_data_from_dataset_generator():
    dataset_generator()


if __name__ == '__main__':
    make_data_from_sources()
    make_data_from_dataset_generator()
