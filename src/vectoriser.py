import argparse
import os
from typing import List, Set

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
from smolagents import CodeAgent, LiteLLMModel, Tool
from tqdm import tqdm
from transformers import AutoTokenizer


def get_data_dir(folder_name):
    root_dir = os.environ['ROOT_DATA_DIR']
    return os.path.join(root_dir, folder_name)

class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )



print('Load dotenv', load_dotenv())
ANTHROPIC_TOKEN = os.environ['ANTHROPIC_API_KEY']

# @dataclass
# class Document:
#     """Document dataclass with id and text fields."""
#     id: str
#     text: str
    
#     def __post_init__(self):
#         """Validate document after initialization."""
#         if self.id is None:
#             self.id = str(uuid4())
#         if not isinstance(self.text, str):
#             raise TypeError("Document text must be a string")

# class SimpleLineSplitter:
#     """Simple text splitter that splits documents by lines."""
    
#     def __init__(self, lines_per_chunk=20):
#         self.lines_per_chunk = lines_per_chunk
    
#     def split_documents(self, documents):
#         """Split documents into chunks of specified lines."""
#         chunks = []
#         for doc in documents:
#             if hasattr(doc, 'page_content'):
#                 text = doc.page_content
#             else:
#                 text = str(doc)
            
#             lines = text.split('\n')
            
#             for i in range(0, len(lines), self.lines_per_chunk):
#                 chunk_lines = lines[i:i + self.lines_per_chunk]
#                 chunk_text = '\n'.join(chunk_lines)
                
#                 if chunk_text.strip():  # Only add non-empty chunks
#                     chunks.append(chunk_text)
        
#         return chunks

# Usage example
# text_splitter = SimpleLineSplitter(lines_per_chunk=20)

tokenizer_name = "thenlper/gte-small"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_folder=get_data_dir('models'))

def get_text_splitter():
    chunk_size = 200
    chunk_overlap = 20

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return text_splitter

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks and remove duplicates"""
    print("Splitting documents...")
    text_splitter = get_text_splitter()
    docs_processed = []
    unique_texts: Set[str] = set()
    
    for doc in tqdm(documents, desc="Chunking documents"):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts.add(new_doc.page_content)
                docs_processed.append(new_doc)
    
    print(f"Created {len(docs_processed)} unique document chunks")
    return docs_processed

def get_raw_knowledge_base(root_dir):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            skip_extensions = ('.jpg', '.ico', '.css', '.png', '.pack', '.idx', 'ttf', 'eot', 'woff', 'rev')
            for file in files:
                if not (
                    file.endswith(skip_extensions) or file.startswith('.') or '.' not in file
                ):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                # print(f"Processing file: {file_path}")
    print(f'files prepared {len(file_paths)}')
    knowledge_base = []
    for file_path  in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                text = file.read()
                knowledge_base.append({
                    "text": text,
                    "source": file_path
                })
            except UnicodeDecodeError:
                print('>>>', file_path)
    return knowledge_base


def create_vector_database(docs_processed: List[Document]) -> FAISS:
    """Create a vector database from processed documents"""
    print("Embedding documents... This may take a few minutes")

    cache_folder = get_data_dir('models')
    embedding_model_name="thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, cache_folder=cache_folder)
    return FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )


if __name__  == '__main__':
    parser = argparse.ArgumentParser(description="Simple CLI application")
    
    parser.add_argument('--input', '-i', required=True, help='Input dir')
    args = parser.parse_args()


    knowledge_base = get_raw_knowledge_base(args.input)
    documents = [
        Document(
            page_content=doc["text"], 
            metadata={"source": doc["source"]}
        ) for doc in knowledge_base
    ]
    chunked_docs = chunk_documents(documents)
    print(f'Raw files {len(knowledge_base)}, docs {len(documents)}, Cnunks {len(chunked_docs)}')
    vector_db = create_vector_database(documents)

    model = LiteLLMModel(
        model="anthropic/claude-3-haiku-20240307",
        api_key=ANTHROPIC_TOKEN,
        temperature=0.7,
        max_tokens=1000
    )
    vectordb = create_vector_database(chunked_docs)
    retriever_tool = RetrieverTool(vectordb)
    # agent = ToolCallingAgent(tools=[retriever_tool], model=model)
    agent = CodeAgent(tools=[retriever_tool], model=model)

    result = agent.run(
        "How to  build customer classification",
    )
