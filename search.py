import sys, chromadb
from dotenv import load_dotenv
from openai import OpenAI
from utilities import get_config

embedmodel = get_config("ollama")["embedmodel"]
mainmodel = get_config("ollama")["mainmodel"]

# Load .env file and store API Key as environment variable
load_dotenv()

# Get OpenAI client
openai_client = OpenAI()

# Get ChromaDB client
chroma = chromadb.HttpClient(host="localhost", port=8000)

# Get ChromaDB collection
collection = chroma.get_or_create_collection("persreg")

#query = " ".join(sys.argv[1:])
query = "Wie ist das wenn ich zum Zahnarzt gehen muss?"
queryembed = ollama.embeddings(model=embedmodel, prompt=query)['embedding']

relevantdocs = collection.query(query_embeddings=[queryembed], n_results=5)["documents"][0]
docs = "\n\n".join(relevantdocs)
modelquery = f"""
You are the responsible HR Assistant of Incore Technology.
Your tasks is to answer questions from employees.
You will strictly follow following rules:
- Keep a formal and professional tone when you answer questions.
- Answers must be short and on point.
- Only use provided information in section "background information" to give answers.
- If you think the back ground information is not useful enough, tell the user, that you don't have enough information to answer the question adequately.

User quers:
{query}

Background information:
 {docs}
"""

stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)

for chunk in stream:
  if chunk["response"]:
    print(chunk['response'], end='', flush=True)