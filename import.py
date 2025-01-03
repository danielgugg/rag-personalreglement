import ollama, chromadb, time
from openai.embeddings_utils import get_embedding
import utilities as util

file_name = "Personalreglement.txt"
collection_name = "Personalreglement"

chroma = chromadb.HttpClient(host="localhost", port=8000)
if any(collection == collection_name for collection in chroma.list_collections()):
    print("Deleting collection")
    chroma.delete_collection(collection_name)
collection = chroma.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

embed_model = util.get_config("ollama")["embedmodel"]

start_time = time.perf_counter()

with open(file_name, encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\n', ' ')

chunks = util.chunk_text_by_sentences(source_text=text, sentences_per_chunk=7, overlap=0, language="german")

print(f"With {len(chunks)} chunks")

for index, chunk in enumerate(chunks):
    embed = util.embed_text(chunk)
    print(".", end="", flush=True)
    collection.add([collection_name+str(index)], embeddings=[embed], documents=[chunk], metadatas={"source" : collection_name})

print("--- %s seconds ---" % (time.perf_counter() - start_time))