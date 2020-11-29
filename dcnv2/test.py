
import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs
"""
import tensorflow_datasets as tfds
from dcn import DCN
ratings = tfds.load("movie_lens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
    "user_gender": int(x["user_gender"]),
    "user_zip_code": x["user_zip_code"],
    "user_occupation_text": x["user_occupation_text"],
    "bucketized_user_age": int(x["bucketized_user_age"]),
})
feature=list(ratings.take(1).as_numpy_iterator())[0]
label=feature.pop('user_rating')
print(label,feature)
#vocab=np.load('vocab.npy',allow_pickle=True)
#vocab=vocab.item()
#model = DCN( deep_layer_sizes=[192, 192],vocab=vocab, projection_dim=None)
#model.load_weights('weights')
#print(model._all_features)
"""
model=tf.keras.models.load_model('weights')
feature = {'movie_id': tf.constant(['357']), 'user_id': tf.constant(['138']), 'user_gender': tf.constant([1]), 'user_zip_code': tf.constant(['53211']), 'user_occupation_text': tf.constant(['doctor']), 'bucketized_user_age': tf.constant([45])}
out=model(feature)
print(out)


