#-*- coding:utf-8 -*-

import numpy as np


def init_output():
    user_id = []
    gender = []
    age = []
    occupation = []
    zip_code = []
    hist_movie_id = []
    pos_movie_id = []
    neg_movie_id = []


    return user_id, gender, age, occupation, zip_code, \
        hist_movie_id, pos_movie_id, neg_movie_id


def file_generator(input_path, batch_size):

    user_id, gender, age, occupation, zip_code, \
        hist_movie_id, pos_movie_id, neg_movie_id = init_output()

    cnt = 0
    
    num_lines = sum([1 for line in open(input_path)])

    while True:

        with open(input_path, 'r') as f:
            for line in f.readlines():

                buf = line.strip().split('\t')

                user_id.append(int(buf[0]))
                gender.append(int(buf[1]))
                age.append(int(buf[2]))
                occupation.append(int(buf[3]))
                zip_code.append(int(buf[4]))
                hist_movie_id.append(np.array([int(i) for i in buf[5].strip().split(",")]))
                pos_movie_id.append(int(buf[7]))
                neg_movie_id.append(np.array([int(i) for i in buf[8].strip().split(",")]))

                cnt += 1

                if cnt % batch_size == 0 or cnt == num_lines:
                    user_id = np.array(user_id, dtype='int32')
                    gender = np.array(gender, dtype='int32')
                    age = np.array(age, dtype='int32')
                    occupation = np.array(occupation, dtype='int32')
                    zip_code = np.array(zip_code, dtype='int32')
                    hist_movie_id = np.array(hist_movie_id, dtype='int32')
                    pos_movie_id = np.array(pos_movie_id, dtype='int32')
                    neg_movie_id = np.array(neg_movie_id, dtype='int32')

                    label = np.zeros(len(user_id)) # pos index: 0, other 10 neg samples index [1-10]

                    yield [user_id, gender, age, occupation, zip_code, hist_movie_id, pos_movie_id, neg_movie_id], label

                    user_id, gender, age, occupation, zip_code, hist_movie_id, pos_movie_id, neg_movie_id = init_output()

