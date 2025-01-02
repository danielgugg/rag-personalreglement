import ollama, chromadb, time
from utilities import get_config

collection_name = "persreg"

chroma = chromadb.HttpClient(host="localhost", port=8000)
if any(collection == collection_name for collection in chroma.list_collections()):
    print("Deleting collection")
    chroma.delete_collection(collection_name)
collection = chroma.get_or_create_collection(name="persreg", metadata={"hnsw:space": "cosine"})

embed_model = get_config()["embedmodel"]
start_time = time.perf_counter()

with open("persreg.txt") as f:
    text = f.read().decode("utf-8")

chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=7, overlap=0)

print(f"With {len(chunks)} chunks")

for index, chunk in enumerate(chunks):
    embed = ollama.embeddings(model=embed_model, prompt=chunk)['embedding']
    print(".", end="", flush=True)
    collection.add(["persreg"+str(index)], [embed], documents=[chunk], metadatas={"source" : "Personalreglement"})

print("--- %s seconds ---" % (time.perf_counter() - start_time))