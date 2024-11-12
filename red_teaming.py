import nest_asyncio
import warnings
import os
from dotenv import load_dotenv
import giskard
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import giskard
import pandas as pd

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

kb_path = "policy_store"

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

policy_doc_location = "docs/pb116349-business-health-select-handbook-1024-pdfa.pdf"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader(policy_doc_location)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

db = FAISS.from_documents(loader.load_and_split(text_splitter), OpenAIEmbeddings())

# Prepare QA chain
PROMPT_TEMPLATE = """You are the insurence policy Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on insurence policies.
You will be given a question and relevant excerpts from the insurence policy documents.
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
policy_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

# Test that everything works
response = policy_qa_chain.invoke({"query": "How much can I claim for optical expenses?"})
print(response)

def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function. The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    return [policy_qa_chain.invoke({"query": question}) for question in df["question"]]

giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Insurence policy document question answering",
    description="This model answers any question about the insurence policy document",
    feature_names=["question"],
)

# Optional: let’s test that the wrapped model works
examples = [
    "Whats the cashback amount for optical expenses?",
    "Whats the cashback amount for dental expenses?",
]
giskard_dataset = giskard.Dataset(pd.DataFrame({"question": examples}), target=None)

print(giskard_model.predict(giskard_dataset).prediction)

full_report = giskard.scan(giskard_model, giskard_dataset)

# save the full_report HTML to a file
with open("scan_report.html", "w") as f:
    f.write(full_report.to_html())
