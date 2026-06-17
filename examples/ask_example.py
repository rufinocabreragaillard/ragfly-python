"""Ejemplo básico del SDK de RAGfly."""
from ragfly import RAGfly

client = RAGfly(api_key="rfly_tu_api_key")

# Pregunta simple (respuesta completa)
resp = client.ask("¿Cuáles son las ventas del Q1?")
print(resp.answer)

# Pregunta con streaming
print("\n--- Streaming ---")
for chunk in client.ask("Resumí los contratos vigentes", stream=True):
    print(chunk.delta, end="", flush=True)
print()

# Búsqueda semántica (solo retrieval, sin generación)
results = client.search("contratos de mantenimiento", limit=5)
print(f"\n{results.total_documentos} documentos encontrados")
for doc in results.documents:
    print(f"  · {doc.nombre} (score: {doc.rrf_score:.3f})")
    for chunk in doc.chunks[:1]:
        print(f"      "{chunk.texto[:120]}…"")
