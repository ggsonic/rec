import tensorflow_hub as hub
import time
import pickle
import os
import annoy


module_url = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1'
index_filename = "index"
embedding_dimension = 64
# Load the TF-Hub module
print("Loading the TF-Hub module...")
start=time.time()
embed_fn = hub.load(module_url)
print('time span:',time.time()-start)
print("TF-Hub module is loaded.")

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

query = "world economic growth"

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




