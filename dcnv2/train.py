
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

from dcn import DCN

#ratings = tfds.load("movie_lens/100k-ratings", split="train")
ratings = tfds.load("movie_lens/1m-ratings", split="train")
ratings = ratings.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
    "user_gender": int(x["user_gender"]),
    "user_zip_code": x["user_zip_code"],
    "user_occupation_text": x["user_occupation_text"],
    "bucketized_user_age": int(x["bucketized_user_age"]),
})

tf.random.set_seed(42)
shuffled = ratings.shuffle(1_000_000, seed=42,reshuffle_each_iteration =False)

train = shuffled.take(800_000)
test = shuffled.skip(800_000).take(200_000)

#cached_train = train.shuffle(100_000).batch(8192).cache()
cached_train = train.shuffle(800_000,reshuffle_each_iteration=True).batch(8192)
cached_test = test.batch(4096).cache()

epochs = 8
learning_rate = 0.01
vocab=np.load('vocab.npy',allow_pickle=True)
vocab=vocab.item()
model = DCN( deep_layer_sizes=[192, 192],vocab=vocab, projection_dim=20)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

model.fit(cached_train, epochs=epochs, verbose=False)
#model.save_weights('weights',save_format='tf')
model.save('weights')
metrics = model.evaluate(cached_test, return_dict=True)
print(metrics)

feature = {'movie_id': tf.constant(['357']), 'user_id': tf.constant(['138']), 'user_gender': tf.constant([1]), 'user_zip_code': tf.constant(['53211']), 'user_occupation_text': tf.constant(['doctor']), 'bucketized_user_age': tf.constant([45])}
out=model(feature)
print(out)
