import chromadb, time
import utilities as util

file_name = util.get_config()["file_name"]
collection_name = util.get_config()["collection_name"]

chroma = chromadb.HttpClient(host="localhost", port=8000)

if any(collection == collection_name for collection in chroma.list_collections()):
    print("Deleting collection")
    chroma.delete_collection(collection_name)
collection = chroma.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

start_time = time.perf_counter()

# Read document
with open(f"./documents/{file_name}", encoding='utf-8') as f:
    text = f.read()
    text = text.replace("\n", " ")

# Make chunks out of the document
chunks = util.chunk_text_by_sentences(source_text=text, sentences_per_chunk=7, overlap=0, language="german")

print(f"With {len(chunks)} chunks")

# Make embedding (vectors) out of chunks and add to ChromaDB
for index, chunk in enumerate(chunks):
    embed = util.get_embedding(chunk)
    print(".", end="", flush=True)
    collection.add(ids=[collection_name+str(index)], embeddings=[embed], documents=[chunk], metadatas={"source" : collection_name})

print("--- %s seconds ---" % (time.perf_counter() - start_time))