from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from my_paths import BGE_M3_MODEL_PATH


def _ensure_model_downloaded(model_path: str, repo_id: str = "BAAI/bge-m3") -> None:
    """Download the model from HuggingFace if not already present locally."""
    path = Path(model_path)
    # Consider the model present if the directory exists and contains at least one .safetensors or .bin file
    already_downloaded = path.is_dir() and any(
        path.glob("*.safetensors")
    ) or (path.is_dir() and any(path.glob("*.bin")))
    if not already_downloaded:
        print(f"BGE-M3 model not found at '{model_path}'. Downloading from HuggingFace ({repo_id})...")
        snapshot_download(repo_id=repo_id, local_dir=model_path)
        print(f"BGE-M3 model downloaded to '{model_path}'.")


class MyAPIEmbeddings(Embeddings):
    """
    Embedding function for ChromaDB using BGE-M3 loaded from a local directory.
    Compatible with LangChain's Embeddings interface.

    The model must be downloaded once and placed at BGE_M3_MODEL_PATH
    (defined in my_paths.py as models/bge-m3). Download with:
        from huggingface_hub import snapshot_download
        snapshot_download("BAAI/bge-m3", local_dir="models/bge-m3")

    Args:
        model_path: Path to the local BGE-M3 model directory.
        chunk_size: Number of texts to embed per batch.
    """

    def __init__(
        self,
        model_path: str = str(BGE_M3_MODEL_PATH),
        chunk_size: int = 256,
    ):
        _ensure_model_downloaded(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.chunk_size = chunk_size

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded)
        # CLS-token pooling (BGE-M3 dense retrieval uses the [CLS] representation)
        embeddings = output.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().float().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.chunk_size):
            batch = texts[i : i + self.chunk_size]
            all_embeddings.extend(self._encode_batch(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._encode_batch([text])[0]
