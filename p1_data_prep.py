import os
import p0_config as cfg
import p0_utilities as util
import pandas as pd
import numpy as np
import csv

'''
Create Directories
'''

try: os.mkdir(cfg.result_path)
except OSError: pass



'''
Check if config parameters are importes correctly
'''

print ("UNKNOWN_WORD - {}".format(cfg.UNKNOWN_WORD))
print ("END_WORD - {}".format(cfg.END_WORD))
print ("NAN_WORD - {}".format(cfg.NAN_WORD))
print ("classes - {}".format(cfg.classes))
print ("data_path - {}".format(cfg.data_path))
print ("train_file - {}".format(cfg.train_file))
print ("test_file - {}".format(cfg.test_file))
print ("embedding_file - {}".format(cfg.embedding_file))
print ("result_path - {}".format(cfg.result_path))

####################
print("Loading train data...")
file_path = os.path.join(cfg.data_path, cfg.train_file)
print ("Train file path - {}".format(file_path))
train_data = pd.read_csv(file_path, error_bad_lines=False)

print("Loading test data...")
file_path = os.path.join(cfg.data_path, cfg.test_file)
print ("Train file path - {}".format(file_path))
#test_data = pd.read_csv(file_path, quoting=csv.QUOTE_NONE)
#test_data = pd.read_csv(file_path, error_bad_lines=False)
test_data = pd.read_csv(file_path)

# Null Handling
list_sentences_train = train_data["comment_text"].fillna(cfg.NAN_WORD).values
list_sentences_test = test_data["comment_text"].fillna(cfg.NAN_WORD).values
y_train = train_data[cfg.classes].values

print("Tokenizing sentences in train set...")
tokenized_sentences_train, words_dict = util.tokenize_sentences(list_sentences_train, {})

print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = util.tokenize_sentences(list_sentences_test, words_dict)

words_dict[cfg.UNKNOWN_WORD] = len(words_dict)
print("Training , Test and Tokenizing done")


#########################
print("Loading embeddings...")
embedding_file_path=os.path.join(cfg.data_path, cfg.embedding_file)

embedding_list, embedding_word_dict = util.read_embedding_list(embedding_file_path)
embedding_size = len(embedding_list[0])
print("Loading embeddings done")


print("Preparing required embeddings...")
embedding_list, embedding_word_dict = util.clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

embedding_word_dict[cfg.UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append([0.] * embedding_size)
embedding_word_dict[cfg.END_WORD] = len(embedding_word_dict)
embedding_list.append([-1.] * embedding_size)

embedding_matrix = np.array(embedding_list)

id_to_word = dict((id, word) for word, id in words_dict.items())

train_list_of_token_ids = util.convert_tokens_to_ids(
    tokenized_sentences_train,
    id_to_word,
    embedding_word_dict,
    cfg.sentences_length)
    
test_list_of_token_ids = util.convert_tokens_to_ids(
    tokenized_sentences_test,
    id_to_word,
    embedding_word_dict,
    cfg.sentences_length)
    
x_train = np.array(train_list_of_token_ids)
x_test = np.array(test_list_of_token_ids)

#### Finished Successfully
from p0_model import get_model

get_model_func = lambda: get_model(
    embedding_matrix,
    cfg.sentences_length,
    cfg.dropout_rate,
    cfg.recurrent_units,
    cfg.dense_size)

from p0_train_utils import train_folds
print("Starting to train models...")
models = train_folds(x_train, y_train, cfg.fold_count, cfg.batch_size, get_model_func)







