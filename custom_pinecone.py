from __future__ import annotations

import logging
import numpy as np
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from typing import Optional, List, Any, Tuple

logging.basicConfig(level=logging.INFO)


class CustomPinecone(Pinecone):

    def similarity_search_with_score(self,
                      dim: int = 1536,
                      k: int = 4,
                      filter: Optional[dict] = None,
                      namespace: Optional[str] = None) -> List[Tuple[Document, float]]:
        if namespace is None:
            namespace = self._namespace
        query_obj = np.random.rand(dim).tolist()
        docs = []
        results = self._index.query(
            [query_obj],
            top_k=k,
            include_metadata=True,
            namespace=namespace,
            filter=filter,
        )
        for res in results["matches"]:
            metadata = res["metadata"]
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                score = res["score"]
                docs.append(
                    (Document(page_content=text, metadata=metadata), score))
            else:
                logging.warning(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )
        return docs
    
    def similarity_search(
        self,
        dim: int = 1536,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return pinecone documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Dictionary of argument(s) to filter on metadata
            namespace: Namespace to search in. Default will search in '' namespace.

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(
            dim=dim, k=k, filter=filter, namespace=namespace, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_existing_index(
        cls,
        index_name: str,
        embedding: Embeddings,
        text_key: str = "text",
        namespace: Optional[str] = None,
    ) -> CustomPinecone:
        """Load pinecone vectorstore from index name."""
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please install it with `pip install pinecone-client`."
            )
        return cls(
            pinecone.Index(
                index_name), embedding.embed_query, text_key, namespace
        )
