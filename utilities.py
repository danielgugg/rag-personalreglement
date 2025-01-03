import configparser
from typing import List
from nltk.tokenize import sent_tokenize
import nltk
#nltk.download('punkt_tab')

def embed_text(text:str):
    return get_embedding(text, engine="text-embedding-ada-002")
    
def get_config(section:str):
    config = configparser.ConfigParser()
    config.read('config.ini')
    return dict(config.items(section))

def chunk_text_by_sentences(source_text: str, sentences_per_chunk: int, overlap: int, language:str="english") -> List[str]:
    """
    Splits text by sentences
    """
    if sentences_per_chunk < 2:
        raise ValueError("The number of sentences per chunk must be 2 or more.")
    if overlap < 0 or overlap >= sentences_per_chunk - 1:
        raise ValueError("Overlap must be 0 or more and less than the number of sentences per chunk.")
    
    sentences = sent_tokenize(source_text, language=language)
    if not sentences:
        print("Nothing to chunk")
        return []
    
    chunks = []
    i = 0
    print(f"Number of sentences: {len(sentences)}")
    while i < len(sentences):
        end = min(i + sentences_per_chunk, len(sentences))
        chunk = ' '.join(sentences[i:end])
        
        if overlap > 0 and i > 1:
            overlap_start = max(0, i - overlap)
            overlap_end = i
            overlap_chunk = ' '.join(sentences[overlap_start:overlap_end])
            chunk = overlap_chunk + ' ' + chunk
        
        chunks.append(chunk.strip())
        i += sentences_per_chunk
    
    return chunks