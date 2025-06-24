import os
from typing import List, Set

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.vectorstores import VectorStore
from smolagents import LiteLLMModel, Tool, ToolCallingAgent
from sweed_rnd.utils import get_data_dir
from tqdm import tqdm
from transformers import AutoTokenizer


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



print(load_dotenv())
ANTHROPIC_TOKEN = os.environ['ANTHROPIC_API_KEY']

cache_folder = get_data_dir('models')
root_dir = '/Users/username/PycharmProjects/ml-research/src/sweed_rnd'
file_paths = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            # print(f"Processing file: {file_path}")
print(f'files prepared {len(file_paths)}')
knowledge_base = []
for file_path  in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        knowledge_base.append({
            "text": text,
            "source": file_path
        })

knowledge_base = get_raw_knowledge_base()
documents = [
    Document(
        page_content=doc["text"], 
        metadata={"source": doc["source"]}
    ) for doc in knowledge_base
]

tokenizer_name = "thenlper/gte-small"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_folder=cache_folder)

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

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks and remove duplicates"""
    print("Splitting documents...")
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

def create_vector_database(docs_processed: List[Document]) -> FAISS:
    """Create a vector database from processed documents"""
    print("Embedding documents... This may take a few minutes")
    embedding_model_name="thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, cache_folder=cache_folder)
    return FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

def main():
    chunked_docs = chunk_documents(documents)
    vectordb = create_vector_database(chunked_docs)

    model = LiteLLMModel(
        model="anthropic/claude-3-haiku-20240307",
        api_key=ANTHROPIC_TOKEN,
        temperature=0.7,
        max_tokens=1000
    )

    retriever_tool = RetrieverTool(vectordb)
    agent = ToolCallingAgent(tools=[retriever_tool], model=model)

    result = agent.run(
        "How to  build customer classification"
    )

    print(result)