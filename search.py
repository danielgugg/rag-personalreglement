import sys, chromadb
from dotenv import load_dotenv
import openai
import os
import utilities as util

# Load .env file and store API Key as environment variable
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Get ChromaDB client
chroma = chromadb.HttpClient(host="localhost", port=8000)

# Get ChromaDB collection
collection_name = util.get_config()["collection_name"]
collection = chroma.get_or_create_collection(collection_name)

systemprompt = """
        You are a helpful HR Assistant at Incore Technology.
        Your responsibilities include responding to HR-related inquiries from the user.
        
        # Guidelines
        Please adhere strictly to the following guidelines when providing answers:
        - Use only the documents provided below as the basis for your responses.
        - Always answer in the language the user writes.
        - Maintain a formal and professional tone in all communications.
        - Ensure answers are concise and directly address the question.
        - If the information cannot be found in the documents or the query is not HR-related, respond accordingly and refer to Incore HR, Elvira Maronas.
        """
conversation_history = [{"role": "system", "content": systemprompt}]

# Main loop for handling user queries
while True:
    # Prompt user for input
    query = input("You: ")
    
    # Exit condition
    if query.lower() in ["exit", "quit"]:
        print("Have a great day!")
        break

    # Generate embeddings for the query
    try:
        # Embedding generation
        queryembed = util.get_embedding(text=query)
        
        # Query ChromaDB for relevant documents
        relevantdocs = collection.query(query_embeddings=[queryembed], n_results=5)["documents"][0]
        docs = "\n\n".join(relevantdocs)
        
        # Create the model prompt
        modelquery = f"""
        # Documents
        \"\"\"
        {docs}
        \"\"\"

        # Question
        {query}
        """

        # Add user input to the conversation history
        conversation_history.append({"role": "user", "content": modelquery})
        
        # Send the query to OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=conversation_history
        )
        
        # Extract the assistant's response
        answer = response.choices[0].message.content

         # Print the assistant's response
        print(f"AI: {answer}")

         # Add the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": answer})
        
    except Exception as e:
            print(f"An error occurred: {e}")