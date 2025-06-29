import argparse
import hashlib
import json
import os

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from vectoriser import (
    VECTOR_INDEX_DIR_NAME,
    chunk_documents,
    create_vector_database,
    get_data_dir,
    get_raw_knowledge_base,
    save_vector_db,
)


def dump_json(input_dict, file_path):
    with open(file_path, "w") as file:
        json.dump(input_dict, file, indent=4)
    print(f'Dumped to {file_path}')

def load_json(file_path):
    res = None
    with open(file_path, "r") as file:
        res = json.load(file)
    print(f'Loaded from {file_path}')
    return res

def load_text(file_path):
    res = None
    with open(file_path, "r") as file:
        res = file.read()
    print(f'Loaded from {file_path}')
    return res

def saved_text(text_input, file_path):
    with open(file_path, "w") as file:
        file.write(text_input)
    print(f'Dumped to {file_path}')

def eval_hash(seed_phrase, length: int = None) -> str:
    res = str(hashlib.md5(seed_phrase.encode('utf-8')).hexdigest())[:length]
    return res

def save_vector_db(vector_db: FAISS, path: str):
    """Save vector database to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vector_db.save_local(path)
    print(f"Vector database saved to: {path}")

def build_index(knowledge_base, vector_db_path):
    """Build or rebuild the knowledge base from input directory"""
    files = os.listdir(vector_db_path)
    if len(files) > 0:
        status = {'Success': True}
        return status
    try:
        documents = [
            Document(
                page_content=doc["text"], 
                metadata={"source": doc["source"]}
            ) for doc in knowledge_base
        ]
        chunked_docs = chunk_documents(documents, models_dir=get_data_dir('models'))
        vector_db = create_vector_database(chunked_docs, models_dir=get_data_dir('models'))
        save_vector_db(vector_db, vector_db_path)
        return {
            "message": "Knowledge base built successfully",
            "status": "success",
            "stats": {
                "raw_files": len(knowledge_base),
                "documents": len(documents),
                "chunks": len(chunked_docs)
            }
        }
    except Exception as e:
        raise RuntimeError(f"Failed to build knowledge base: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple CLI application")
    
    parser.add_argument('--input', '-i', required=True, help='Input dir')
    args = parser.parse_args()


    knowledge_base = get_raw_knowledge_base(args.input)
    base_whole_text = ','.join([i["text"] for i in knowledge_base])
    index_name = eval_hash(base_whole_text)
    print(len(base_whole_text))
    index_name = eval_hash(base_whole_text)
    print(index_name)

    vector_db_root = get_data_dir(VECTOR_INDEX_DIR_NAME)
    vector_db_path = os.path.join(vector_db_root, index_name)
    index_build_result = build_index(knowledge_base, vector_db_path)

    index_meta_file_path = os.path.join(get_data_dir('plaintext_index'), f'meta_{index_name}.json')
    dump_json(knowledge_base, index_meta_file_path)
    knowledge_base_file_path = os.path.join(get_data_dir('plaintext_index'), f'{index_name}.json')
    dump_json(knowledge_base, knowledge_base_file_path)
    print(index_build_result)