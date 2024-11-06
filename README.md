# carter-ai
Your AI Coach Carter for Technical Interviews

## Developer Quickstart
1.  Create a virtual python environment
    ```bash
    $ conda create -n venv-carter python=3.11
    $ conda activate venv-carter
    ```
2.  Install `poetry`
    ```bash
    $ pip install poetry
    ```
3.  Install dependencies
    ```bash
    $ poetry install
    ```
4. [Download Pre-trained LLMs](#Download-Pre-trained-LLMs)

5. Run Example Scripts
    ```bash
    $  torchrun scripts/example_chat_completion.py </path/to/.llama/checkpoints/MODEL-ID>
    ```

## Download Pre-trained LLMs
See [Meta Llama Download Instrcutions](https://github.com/meta-llama/llama-models/tree/main?tab=readme-ov-file#download)

## 