# AIMOJI

A learning project to create a model to suggest an emoji at a cursor position in a text editor.

## Prerequisites
#### Install `uv`
[Installation | `uv`](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) 

#### Configure Kaggle
Go to [Kaggle settings](https://www.kaggle.com/settings) and click "Create New Token".

**Linux/macOS**:
```sh
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Windows**:
Move the downloaded file to `C:\Users\<Windows-username>\.kaggle\kaggle.json`.

## Usage
#### Install dependencies
```sh
uv sync
```

#### Download the raw data
Requires [Kaggle configuration](#configure-kaggle)
```sh
uv run src/download.py
```

---

Training data format:
```jsonl
{"before": "Do not drink ", "prediction": "☕", "after": " after lunch"}
```
But in `.parquet` files

Model inputs, `sentence-transformers/all-MiniLM-L6-v2` embeddings of `before` and `after`.
Model outputs -> one hot or top 3 predictions.

```
Text -> sentence-transformers/all-MiniLM-L6-v2 encoder -> logistic regression classifier -> prediction
```

Raw Data chosen for the project is `./data/emojitweets-01-04-2018.txt`. Each line is a cleaned up english tweet with emojis
