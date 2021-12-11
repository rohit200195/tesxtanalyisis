import transformers

DEVICE = "cpu"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
# BERT_PATH = "bert-base-uncased"
# MODEL_PATH = "/root/docker_data/model.bin"
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH = "model.bin"

TRAINING_FILE = "abhishekthakur.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
