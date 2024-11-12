import warnings
import os
import pandas as pd
import giskard
import os
from dotenv import load_dotenv
from giskard.rag import KnowledgeBase, generate_testset, QATestset
from rag_pipeline import get_nodes_from_pdf
from giskard.llm.client.openai import OpenAIClient

warnings.filterwarnings('ignore')
_ = load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

giskard.llm.set_llm_api("openai")
gpt4o_mini = OpenAIClient(model="gpt-4o-mini")
giskard.llm.set_default_client(gpt4o_mini)

def get_knowledge_base(pdf_path):
    nodes = get_nodes_from_pdf(pdf_path)
    knowledge_base_df = pd.DataFrame([node.text for node in nodes], columns=["text"])
    knowledge_base = KnowledgeBase(knowledge_base_df, llm_client = giskard.llm.set_default_client(gpt4o_mini))
    return knowledge_base

def get_test_set(pdf_path):
    knowledge_base = get_knowledge_base(pdf_path)
    testset = generate_testset(knowledge_base,
                            num_questions=6,
                            agent_description="A chatbot answering questions about the insurence policy document")
    return testset

def get_test_set_df(pdf_path):
    testset = get_test_set(pdf_path)
    df_testset = testset.to_pandas()
    print(df_testset)
    print(df_testset["metadata"].apply(lambda x: x["question_type"]).unique())
    df_testset.to_csv("test_set.csv", index=False)
    return df_testset

# test
# pdf_path = "docs/pb116349-business-health-select-handbook-1024-pdfa.pdf"
# get_test_set_df(pdf_path)
     
