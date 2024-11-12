import warnings
from dotenv import load_dotenv
from giskard.rag import evaluate, RAGReport
from giskard.rag.metrics.ragas_metrics import ragas_context_recall, ragas_context_precision, ragas_faithfulness, ragas_answer_relevancy
from rag_pipeline import get_query_engine
from test_set_gen import get_test_set, get_knowledge_base

warnings.filterwarnings('ignore')
_ = load_dotenv()

policy_store_path = "policy_store"

query_engine = get_query_engine(policy_store_path)

pdf_path = "docs/pb116349-business-health-select-handbook-1024-pdfa.pdf"

testset = get_test_set(pdf_path)

knowledge_base = get_knowledge_base(pdf_path)

def answer_fn(question):
    answer = query_engine.query(question)
    return str(answer)

report = evaluate(answer_fn,
                testset=testset,
                knowledge_base=knowledge_base,
                metrics=[ragas_context_recall, ragas_context_precision, ragas_faithfulness, ragas_answer_relevancy])

report_df = report.to_pandas()
print(report_df)
# save to csv
report_df.to_csv("report.csv", index=False)

