from schema.fk_graph import FKGraph
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import setup_logger
logger = setup_logger("refiner")

def build_table_document(table_name:str, info:dict)->str: 
    result = [fk["from"]+" refrences "+fk["to_table"] +"."+ fk["to_column"] for fk in info["foreign_keys"]]
    t1=" ".join(info['columns']) 
    temp = " ".join(str(v) for v in result) 
    s =f" {table_name}: {t1} foreign_keys: {temp}"
    return s

def build_faiss_index(full_schema: dict): 
    x=[build_table_document(table_name,info) for table_name,info in full_schema.items()] 
    tables = [table_name for table_name in list(full_schema.keys())]
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 
    vectorstore=FAISS.from_texts(x, embedding_model)
    fk_graph = FKGraph(full_schema)
    
    return vectorstore, tables, fk_graph


def refine_schema(
    faiss:FAISS,
    full_schema: dict,
    user_query: str,
    fk_graph:FKGraph,
    top_k: int = 3,
    fk_hops: int = 1
) -> dict:
    """
    Dynamically selects the most relevant subset of the schema.

    Args:
        full_schema (dict): Cached full schema
        user_query (str): User question
        top_k (int): Number of seed tables
        fk_hops (int): FK graph expansion depth

    Returns:
        dict: Refined schema
    """

    keywords = user_query.lower().split()
    docs = faiss.similarity_search(user_query, k=top_k) # Extract table name from each result
    seed_tables = [doc.page_content.split(":")[0].strip() for doc in docs]
    # Expand via foreign keys

    expanded_tables = fk_graph.expand_tables(seed_tables, hops=fk_hops)
    # Column pruning: keep only relevant columns
    refined_schema = {}
    for table in expanded_tables:
        info = full_schema[table]
        cols = info.get("columns", [])

        relevant_cols = [
            col for col in cols
            if col.lower() in keywords or table in seed_tables
        ]

        # Fallback: keep all columns if pruning removes everything
        if not relevant_cols:
            relevant_cols = cols

        refined_schema[table] = {
            **info,
            "columns": relevant_cols
        }
    logger.info(f"seed_tables for query '{user_query}': {seed_tables}")
    logger.info(f"expanded_tables after FK: {list(refined_schema.keys())}")
    return refined_schema
