import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "/Users/matt/Coding/bert-ner-tutorial/input/bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "/Users/matt/Coding/bert-ner-tutorial/input/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True,
)

# TO DO: Note the absolute paths above, how embarrassing! I'm going to leave for now rather than try to dig in to the
# incricacies of hugging face - there seems to be a mismatch between the tutorial and how transformers works now?

""" 
For reference:
When trying to train with relative paths as shown in the video, we get:

matt$ python src/train.py
Traceback (most recent call last):
  File "/Users/matt/Coding/bert-ner-tutorial/src/train.py", line 12, in <module>
    import config
  File "/Users/matt/Coding/bert-ner-tutorial/src/config.py", line 10, in <module>
    TOKENIZER = transformers.BertTokenizer.from_pretrained(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/matt/Coding/bert-ner-tutorial/env/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 1771, in from_pretrained
    resolved_vocab_files[file_id] = cached_file(
                                    ^^^^^^^^^^^^
  File "/Users/matt/Coding/bert-ner-tutorial/env/lib/python3.11/site-packages/transformers/utils/hub.py", line 417, in cached_file
    resolved_file = hf_hub_download(
                    ^^^^^^^^^^^^^^^^
  File "/Users/matt/Coding/bert-ner-tutorial/env/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 112, in _inner_fn
    validate_repo_id(arg_value)
  File "/Users/matt/Coding/bert-ner-tutorial/env/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 160, in validate_repo_id
    raise HFValidationError(
huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bert-base-uncased'. Use `repo_type` argument if needed.

 """
