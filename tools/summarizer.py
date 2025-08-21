"""
Summarizer Tool for generating concise summaries of the entire report
Uses the full document context for comprehensive summaries
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SummarizerTool:
    """Tool for generating document summaries"""

    def __init__(self, vector_store, llm_model):
        """
        Initialize summarizer tool

        Args:
            vector_store: Vector store containing all document chunks
            llm_model: LLM for summary generation
        """
        self.vector_store = vector_store
        self.llm_model = llm_model

    async def execute(self, query: str = None) -> Dict[str, Any]:
        """
        Execute summarization

        Args:
            query: Optional specific focus for summary

        Returns:
            Dictionary with summary and metadata
        """
        try:
            logger.info("Executing document summarization...")

            # Get all document content for comprehensive summary
            # Use a broad query to get representative chunks
            search_query = query if query else "market analysis competitive landscape SWOT financial performance"
            search_results = self.vector_store.search(search_query, top_k=min(8, self.vector_store.size))

            if not search_results:
                return {
                    "answer": "No content available for summarization.",
                    "metadata": {"error": "No documents found"}
                }

            # Combine all chunks for full context
            full_context = "\n\n".join([result['content'] for result in search_results])

            # Generate summary
            system_prompt = """You are an expert business analyst specializing in market research reports. 
Create a comprehensive yet concise summary of the Innovate Inc. market research report.

Guidelines:
- Cover all key sections: company overview, market size, competition, SWOT analysis, and conclusions
- Use bullet points for key facts and figures
- Highlight the most important insights and strategic implications
- Keep the summary informative but digestible (3-5 paragraphs)
- Include specific numbers and percentages where mentioned"""

            user_prompt = f"""Please summarize this Innovate Inc. market research report:

{full_context}

Provide a comprehensive summary covering all major sections and key insights."""

            llm_response = await self.llm_model.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.2,  # Slightly higher for more natural flow
                    "num_predict": 500   # Longer output for comprehensive summary
                }
            )

            return {
                "answer": llm_response["response"],
                "metadata": {
                    "content_chunks_used": len(search_results),
                    "total_content_length": len(full_context),
                    "summary_type": "comprehensive" if not query else "focused",
                    "focus_area": query if query else "all_sections",
                    "generation_time": llm_response.get("total_duration", 0),
                    "tokens_generated": llm_response.get("eval_count", 0)
                }
            }

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                "answer": f"An error occurred during summarization: {str(e)}",
                "metadata": {"error": True}
            }
