UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

probabilities_normalize_coefficient = 1.4

data_path = 'data'
train_file = 'train_104505.csv'  #'train.csv'
test_file = 'test.csv' #'test_105706.csv'
embedding_file = 'wiki-news-300d-1M-62499.vec' #'wiki-news-300d-1M.vec'

result_path = 'toxic_results'

batch_size = 256
sentences_length = 500
recurrent_units = 64
dropout_rate = 0.3
dense_size = 32
fold_count = 5 # 10

