import warnings
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
import nest_asyncio;
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pdf_path = "docs/pb116349-business-health-select-handbook-1024-pdfa.pdf"
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm_gpt4o = OpenAI(model="gpt-4o-mini", api_key = OPENAI_API_KEY)
splitter = SentenceSplitter(chunk_size=1024)
path = "policy_store"
parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown")

Settings.llm = llm_gpt4o
Settings.embed_model = embed_model

def get_nodes_from_pdf(pdf_path):
    documents = parser.load_data(pdf_path)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def ingest_pdf(pdf_path, embed_model, path):
    nodes = get_nodes_from_pdf(pdf_path)
    vector_index = VectorStoreIndex(nodes, embed_model = embed_model)
    vector_index.storage_context.persist(persist_dir=path)
    query_engine_gpt4o = vector_index.as_query_engine(similarity_top_k=3, llm=llm_gpt4o)
    return query_engine_gpt4o

def get_query_engine(kb_path):
    ctx = StorageContext.from_defaults(persist_dir=kb_path)
    index = load_index_from_storage(ctx)
    query_engine = index.as_query_engine(similarity_top_k=3, llm=llm_gpt4o)
    return query_engine

# test
# query_engine_gpt4o = ingest_pdf(pdf_path, embed_model, path)
# query = "whats the cashback amount for dental treatments?"
# resp = query_engine_gpt4o.query(query)
# print(str(resp))
    
