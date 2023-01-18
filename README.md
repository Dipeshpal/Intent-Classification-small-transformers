# Zero-shot-finetuning-transformers
 Zero-shot-finetuning-transformers


## What can you do?

- You can create custom classification dataset 
- You can push your custom dataset to huggingface hub
- You can fine-tune any model on your custom dataset (zero-shot), classification task
- You can evaluate your model on your custom dataset
- You can use your model for inference

## How to use?

- Install requirements

  ```commandline
  pip install -r requirements.txt
  ```

### 1. Create custom dataset

- First check 'data.py' file. If you want you can add examples in it. Do not change the format of the examples.

    ```commandline
    python dataset_generator.py
    ```
  
- This will create a dataset in 'data' folder. You can check the dataset in 'dataset_folder' folder.

### 2. Push your dataset to huggingface hub

- You can push your dataset to huggingface hub. You can check the dataset in huggingface hub.

    ```commandline
    python push_to_hub.py
    ```
  
- This will push your dataset to huggingface hub. You can check the dataset in huggingface hub.

### 3. Fine-tune any model on your custom dataset (zero-shot), classification task

- You can fine-tune any model on your custom dataset (zero-shot), classification task. You can check the dataset in huggingface hub.

    ```commandline
    python zero-shot-finetuning-train.py
    ```
  
  or
- You can also check out jupyter notebook 'zero-shot-finetuning-train.ipynb' for more details.
  
- This will fine-tune any model on your custom dataset (zero-shot), classification task. You can check the dataset in huggingface hub.
- This will also save the model and push to huggingface hub.
- You can check the model in huggingface hub.


### 4. Inference

- It is recommended to open the script 'inference.py' and change the model name and dataset name to your model and dataset name.

- Run the script 'inference.py' to get the inference.

    ```commandline
    python inference.py
    ```
  

## How to contribute?

- You can contribute to this repository by adding more examples in 'data.py' file.

## References
https://github.com/huggingface/setfit/blob/main/notebooks/zero-shot-classification.ipynb


## Thanks me on-
Follow me on Instagram: https://www.instagram.com/dipesh_pal17

Subscribe me on YouTube: https://www.youtube.com/dipeshpal17

Donate me: https://www.buymeacoffee.com/dipeshpal
