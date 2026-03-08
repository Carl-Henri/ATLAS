"""
LangChain-compatible embedding wrapper around BGE-M3 for use with RAGAS.
Exposes MyAPIEmbedding (singular), which is an alias for MyAPIEmbeddings.
"""

from bge_m3_embeddings import MyAPIEmbeddings as MyAPIEmbedding

__all__ = ["MyAPIEmbedding"]
