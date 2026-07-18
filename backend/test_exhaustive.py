from app.rag.store import ReportStore

store = ReportStore()
data = store.collection.get(include=['documents', 'metadatas'])

term = 'nano bubble'
hits = []
for doc, meta in zip(data['documents'], data['metadatas']):
    if term.lower() in doc.lower():
        hits.append((meta.get('report_id'), doc[:200]))

print(f'{len(hits)} chunk(s) contain \"{term}\"')
for report_id, snippet in hits:
    print('---', report_id)
    print(snippet)
