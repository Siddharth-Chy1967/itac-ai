from app.rag.store import ReportStore
from app.rag.embeddings import Embedder

store = ReportStore()
embedder = Embedder()
q = embedder.encode_query('roof top coating')
res = store.collection.query(query_embeddings=[q], n_results=5)
for i, doc in enumerate(res['documents'][0]):
    print('---', res['metadatas'][0][i].get('report_id'), res['distances'][0][i])
    print(doc[:200])
