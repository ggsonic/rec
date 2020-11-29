#-*- coding:utf-8 -*-

# https://github.com/shenweichen/DeepMatch/blob/master/examples/colab_MovieLen1M_YoutubeDNN.ipynb


#! wget http://files.grouplens.org/datasets/movielens/ml-1m.zip -O ./ml-1m.zip 
#! wget https://raw.githubusercontent.com/shenweichen/DeepMatch/master/examples/preprocess.py -O preprocess.py
#! unzip -o ml-1m.zip 


import pandas as pd
import numpy as np
from tqdm import tqdm
import random

from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

OUT_TRAIN_FILE='train.txt'
OUT_TEST_FILE='test.txt'

def data_set(data, negsample=0):

    #sort by timestamp to construct history item ids
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]),rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1]),rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]),len(test_set[0]))

    return train_set,test_set

def data_array(train_set,user_profile,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


# 1. Load data
user_key = ['user_id','gender','age','occupation','zip']
user = pd.read_csv('ml-1m/users.dat',sep='::',header=None,names=user_key)

rating_key = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,names=rating_key)

item_key = ['movie_id','title','genres']
movies = pd.read_csv('ml-1m/movies.dat',sep='::',header=None,names=item_key)

data = pd.merge(pd.merge(ratings, movies), user)


print(data.shape)
# (1000209, 10)


# 2. Label Encoding for sparse features, 
# and process sequence features with `date_set` and `data_array`

sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
SEQ_LEN = 50
negsample = 0


features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
feature_max_idx = {}

for feature in features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
item_profile = data[["movie_id"]].drop_duplicates('movie_id')

user_profile.set_index("user_id", inplace=True)
user_item_list = data.groupby("user_id")['movie_id'].apply(list)

train_set, test_set = data_set(data, negsample)
train_model_input, train_label = data_array(train_set, user_profile, SEQ_LEN)
test_model_input, test_label = data_array(test_set, user_profile, SEQ_LEN)


# 3. Create neg samples

train_neg_sample_list = []
test_neg_sample_list = []
all_movie_list = set(data['movie_id'])
neg_sample_num = 10

for i in tqdm(range(len(train_label))):
    a = set(train_model_input['hist_movie_id'][i] + train_model_input['movie_id'][i])
    neg_list = random.sample(list(all_movie_list - a), neg_sample_num)
    train_neg_sample_list.append(np.array(neg_list))
    
for i in tqdm(range(len(test_label))):
    a = set(test_model_input['hist_movie_id'][i] + test_model_input['movie_id'][i])
    neg_list = random.sample(list(all_movie_list - a), neg_sample_num)
    test_neg_sample_list.append(np.array(neg_list))


# 4. Write to .txt

train = open(OUT_TRAIN_FILE, "w")

for i in range(len(train_label)):
    a = train_model_input["user_id"][i]
    b = train_model_input["gender"][i]
    c = train_model_input["age"][i]
    d = train_model_input["occupation"][i]
    e = train_model_input["zip"][i]
    f = train_model_input["hist_movie_id"][i]
    g = train_model_input["hist_len"][i]
    
    h = train_model_input["movie_id"][i]
    m = train_neg_sample_list[i]
    
    train.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"\
               %(str(a), str(b), str(c), str(d), str(e), ','.join([str(ii) for ii in f]), str(g), str(h), ','.join([str(ii) for ii in m])))
    
train.close()



test = open(OUT_TEST_FILE, "w")

for i in range(len(test_label)):
    a = test_model_input["user_id"][i]
    b = test_model_input["gender"][i]
    c = test_model_input["age"][i]
    d = test_model_input["occupation"][i]
    e = test_model_input["zip"][i]
    f = test_model_input["hist_movie_id"][i]
    g = test_model_input["hist_len"][i]
    
    h = test_model_input["movie_id"][i]
    m = test_neg_sample_list[i]
    
    test.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"\
               %(str(a), str(b), str(c), str(d), str(e), ','.join([str(ii) for ii in f]), str(g), str(h), ','.join([str(ii) for ii in m])))
    
test.close()
