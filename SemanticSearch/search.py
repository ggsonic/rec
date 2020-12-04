
import os
import sys
import pickle
from collections import namedtuple
from datetime import datetime
import numpy as np
import apache_beam as beam
from apache_beam.transforms import util
import tensorflow as tf
import tensorflow_hub as hub
import annoy
from sklearn.random_projection import gaussian_random_matrix
import time

embed_fn = None

def generate_embeddings(text, module_url, random_projection_matrix=None):
  # Beam will run this function in different processes that need to
  # import hub and load embed_fn (if not previously loaded)
  global embed_fn
  if embed_fn is None:
    embed_fn = hub.load(module_url)
  embedding = embed_fn(text).numpy()
  if random_projection_matrix is not None:
    embedding = embedding.dot(random_projection_matrix)
  return text, embedding


def to_tf_example(entries):
  examples = []

  text_list, embedding_list = entries
  for i in range(len(text_list)):
    text = text_list[i]
    embedding = embedding_list[i]

    features = {
        'text': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[text.encode('utf-8')])),
        'embedding': tf.train.Feature(
            float_list=tf.train.FloatList(value=embedding.tolist()))
    }

    example = tf.train.Example(
        features=tf.train.Features(
            feature=features)).SerializeToString(deterministic=True)

    examples.append(example)

  return examples

def run_hub2emb(args):
  '''Runs the embedding generation pipeline'''

  options = beam.options.pipeline_options.PipelineOptions(**args)
  args = namedtuple("options", args.keys())(*args.values())

  with beam.Pipeline(args.runner, options=options) as pipeline:
    (
        pipeline
        | 'Read sentences from files' >> beam.io.ReadFromText(
            file_pattern=args.data_dir)
        | 'Batch elements' >> util.BatchElements(
            min_batch_size=args.batch_size, max_batch_size=args.batch_size)
        | 'Generate embeddings' >> beam.Map(
            generate_embeddings, args.module_url, args.random_projection_matrix)
        | 'Encode to tf example' >> beam.FlatMap(to_tf_example)
        | 'Write to TFRecords files' >> beam.io.WriteToTFRecord(
            file_path_prefix='{}/emb'.format(args.output_dir),
            file_name_suffix='.tfrecords')
    )


def generate_random_projection_weights(original_dim, projected_dim):
  random_projection_matrix = None
  random_projection_matrix = gaussian_random_matrix(
      n_components=projected_dim, n_features=original_dim).T
  print("A Gaussian random weight matrix was creates with shape of {}".format(random_projection_matrix.shape))
  print('Storing random projection matrix to disk...')
  with open('random_projection_matrix', 'wb') as handle:
    pickle.dump(random_projection_matrix, 
                handle, protocol=pickle.HIGHEST_PROTOCOL)

  return random_projection_matrix

module_url = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1'
projected_dim = 64

import tempfile

output_dir = tempfile.mkdtemp()
original_dim = hub.load(module_url)(['']).shape[1]
random_projection_matrix = None

if projected_dim:
  random_projection_matrix = generate_random_projection_weights(
      original_dim, projected_dim)

args = {
    'job_name': 'hub2emb-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S')),
    'runner': 'DirectRunner',
    'batch_size': 1024,
    'data_dir': 'corpus/*.txt',
    'output_dir': output_dir,
    'module_url': module_url,
    'random_projection_matrix': random_projection_matrix,
}

print("Pipeline args are set.")
args

print("Running pipeline...")
start=time.time()
run_hub2emb(args)
print('time span:',time.time()-start)
print("Pipeline is done.")


embed_file = os.path.join(output_dir, 'emb-00000-of-00001.tfrecords')
sample = 5

# Create a description of the features.
feature_description = {
    'text': tf.io.FixedLenFeature([], tf.string),
    'embedding': tf.io.FixedLenFeature([projected_dim], tf.float32)
}

def _parse_example(example):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example, feature_description)

dataset = tf.data.TFRecordDataset(embed_file)
for record in dataset.take(sample).map(_parse_example):
  print("{}: {}".format(record['text'].numpy().decode('utf-8'), record['embedding'].numpy()[:10]))



def build_index(embedding_files_pattern, index_filename, vector_length,
    metric='angular', num_trees=100):
  '''Builds an ANNOY index'''

  annoy_index = annoy.AnnoyIndex(vector_length, metric=metric)
  # Mapping between the item and its identifier in the index
  mapping = {}

  embed_files = tf.io.gfile.glob(embedding_files_pattern)
  num_files = len(embed_files)
  print('Found {} embedding file(s).'.format(num_files))

  item_counter = 0
  for i, embed_file in enumerate(embed_files):
    print('Loading embeddings in file {} of {}...'.format(i+1, num_files))
    dataset = tf.data.TFRecordDataset(embed_file)
    for record in dataset.map(_parse_example):
      text = record['text'].numpy().decode("utf-8")
      embedding = record['embedding'].numpy()
      mapping[item_counter] = text
      annoy_index.add_item(item_counter, embedding)
      item_counter += 1
      if item_counter % 100000 == 0:
        print('{} items loaded to the index'.format(item_counter))

  print('A total of {} items added to the index'.format(item_counter))

  print('Building the index with {} trees...'.format(num_trees))
  annoy_index.build(n_trees=num_trees)
  print('Index is successfully built.')

  print('Saving index to disk...')
  annoy_index.save(index_filename)
  print('Index is saved to disk.')
  print("Index file size: {} GB".format(
    round(os.path.getsize(index_filename) / float(1024 ** 3), 2)))
  annoy_index.unload()

  print('Saving mapping to disk...')
  with open(index_filename + '.mapping', 'wb') as handle:
    pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print('Mapping is saved to disk.')
  print("Mapping file size: {} MB".format(
    round(os.path.getsize(index_filename + '.mapping') / float(1024 ** 2), 2)))



embedding_files = "{}/emb-*.tfrecords".format(output_dir)
embedding_dimension = projected_dim
index_filename = "index"

start=time.time()
build_index(embedding_files, index_filename, embedding_dimension)
print('time span:',time.time()-start)

index = annoy.AnnoyIndex(embedding_dimension)
index.load(index_filename, prefault=True)
print('Annoy index is loaded.')
with open(index_filename + '.mapping', 'rb') as handle:
  mapping = pickle.load(handle)
print('Mapping file is loaded.')

def find_similar_items(embedding, num_matches=5):
  '''Finds similar items to a given embedding in the ANN index'''
  ids = index.get_nns_by_vector(
  embedding, num_matches, search_k=-1, include_distances=False)
  items = [mapping[i] for i in ids]
  return items


# Load the TF-Hub module
print("Loading the TF-Hub module...")
start=time.time()
embed_fn = hub.load(module_url)
print('time span:',time.time()-start)
print("TF-Hub module is loaded.")

random_projection_matrix = None
if os.path.exists('random_projection_matrix'):
  print("Loading random projection matrix...")
  with open('random_projection_matrix', 'rb') as handle:
    random_projection_matrix = pickle.load(handle)
  print('random projection matrix is loaded.')

def extract_embeddings(query):
  '''Generates the embedding for the query'''
  query_embedding =  embed_fn([query])[0].numpy()
  if random_projection_matrix is not None:
    query_embedding = query_embedding.dot(random_projection_matrix)
  return query_embedding

extract_embeddings("Hello Machine Learning!")[:10]


query = "confronting global challenges"

print("Generating embedding for the query...")
start=time.time()
query_embedding = extract_embeddings(query)
print('time span:',time.time()-start)

print("")
print("Finding relevant items in the index...")
start=time.time()
items = find_similar_items(query_embedding, 10)
print('time span:',time.time()-start)

print("")
print("Results:")
print("=========")
for item in items:
  print(item)




