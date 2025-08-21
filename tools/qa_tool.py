"""
Q&A Tool for answering specific questions about the market report
Uses retrieval-augmented generation with context from vector store
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class QATool:
    """Tool for answering specific questions using RAG"""

    def __init__(self, vector_store, llm_model):
        """
        Initialize Q&A tool

        Args:
            vector_store: Vector store for document retrieval
            llm_model: LLM for answer generation
        """
        self.vector_store = vector_store
        self.llm_model = llm_model

    async def execute(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Execute Q&A query

        Args:
            query: User question
            top_k: Number of context chunks to retrieve

        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Executing Q&A query: {query[:50]}...")

            # 1. Retrieve relevant context
            search_results = self.vector_store.search(query, top_k=top_k)

            if not search_results:
                return {
                    "answer": "I don't have enough information to answer this question based on the provided document.",
                    "metadata": {
                        "context_chunks": 0,
                        "confidence": 0.0
                    }
                }

            # 2. Build context from top results
            context_chunks = []
            total_score = 0
            for result in search_results:
                context_chunks.append(result['content'])
                total_score += result['score']

            context = "\n\n".join(context_chunks)
            avg_relevance = total_score / len(search_results)

            # 3. Generate answer using LLM
            system_prompt = """You are an expert business analyst. Answer questions accurately based on the provided context from the Innovate Inc. market research report.

Guidelines:
- Only use information from the provided context
- Be specific and cite relevant details
- If the context doesn't contain the answer, say so clearly
- Keep answers concise but informative
- Use bullet points for multiple items when appropriate"""

            user_prompt = f"""Context from Innovate Inc. Market Research Report:

{context}

Question: {query}

Please provide a clear, accurate answer based on the context above."""

            llm_response = await self.llm_model.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.1,  # Low temperature for factual accuracy
                    "num_predict": 300   # Reasonable answer length
                }
            )

            return {
                "answer": llm_response["response"],
                "metadata": {
                    "context_chunks": len(search_results),
                    "avg_relevance_score": round(avg_relevance, 3),
                    "context_preview": context[:200] + "..." if len(context) > 200 else context,
                    "generation_time": llm_response.get("total_duration", 0),
                    "tokens_generated": llm_response.get("eval_count", 0)
                }
            }

        except Exception as e:
            logger.error(f"Q&A execution failed: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "metadata": {"error": True}
            }
