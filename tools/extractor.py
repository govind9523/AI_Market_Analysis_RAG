"""
Data Extractor Tool for extracting structured data in JSON format
Specializes in extracting financial data, market metrics, and key figures
"""

from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class ExtractorTool:
    """Tool for extracting structured data as JSON"""

    def __init__(self, vector_store, llm_model):
        """
        Initialize data extractor tool

        Args:
            vector_store: Vector store for document retrieval
            llm_model: LLM for data extraction
        """
        self.vector_store = vector_store
        self.llm_model = llm_model

        # Predefined extraction schemas for common requests
        self.schemas = {
            "financial": {
                "current_market_size": "string",
                "projected_cagr": "string", 
                "projected_market_size": "string",
                "company_market_share": "string"
            },
            "competitive": {
                "innovate_inc_share": "string",
                "synergy_systems_share": "string",
                "futureflow_share": "string",
                "quantumleap_share": "string"
            },
            "swot": {
                "strengths": "array of strings",
                "weaknesses": "array of strings",
                "opportunities": "array of strings", 
                "threats": "array of strings"
            }
        }

    def _detect_extraction_type(self, query: str) -> Optional[str]:
        """Detect what type of data to extract based on query"""
        query_lower = query.lower()

        if any(term in query_lower for term in ["financial", "market size", "cagr", "revenue", "growth"]):
            return "financial"
        elif any(term in query_lower for term in ["competitive", "competitor", "market share", "competition"]):
            return "competitive"
        elif "swot" in query_lower:
            return "swot"
        else:
            return None

    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute data extraction

        Args:
            query: Extraction request (e.g., "Extract key financial data")

        Returns:
            Dictionary with extracted JSON data and metadata
        """
        try:
            logger.info(f"Executing data extraction: {query[:50]}...")

            # 1. Detect extraction type and get relevant context
            extraction_type = self._detect_extraction_type(query)

            if extraction_type:
                # Use targeted search based on extraction type
                if extraction_type == "financial":
                    search_query = "market size growth CAGR billion revenue financial"
                elif extraction_type == "competitive":
                    search_query = "market share competitors Synergy FutureFlow QuantumLeap percentage"
                elif extraction_type == "swot":
                    search_query = "strengths weaknesses opportunities threats SWOT analysis"
                else:
                    search_query = query
            else:
                search_query = query

            search_results = self.vector_store.search(search_query, top_k=5)

            if not search_results:
                return {
                    "answer": json.dumps({"error": "No relevant data found for extraction"}),
                    "metadata": {"extraction_successful": False}
                }

            # 2. Build context
            context = "\n\n".join([result['content'] for result in search_results])

            # 3. Prepare extraction prompt with schema
            schema_info = ""
            if extraction_type and extraction_type in self.schemas:
                schema_info = f"\nExpected JSON schema: {json.dumps(self.schemas[extraction_type], indent=2)}"

            system_prompt = f"""You are a data extraction specialist. Extract structured information from the provided text and return it as valid JSON.

Guidelines:
- Return ONLY valid JSON, no additional text or explanations
- Use exact values from the text when available
- Use "null" for missing information
- Ensure all percentage values include the % symbol
- Ensure all monetary values include currency symbols ($, billion, etc.)
- For arrays, extract individual items as separate strings{schema_info}

Focus on extracting: {query}"""

            user_prompt = f"""Extract structured data from this Innovate Inc. market research report:

{context}

Extraction request: {query}

Return the extracted data as valid JSON:"""

            # 4. Generate JSON response
            llm_response = await self.llm_model.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=self.schemas.get(extraction_type)
            )

            # 5. Format response
            if llm_response.get("is_valid_json") and llm_response.get("parsed_json"):
                extracted_data = llm_response["parsed_json"]
                json_response = json.dumps(extracted_data, indent=2)
            else:
                # Fallback: return raw response if JSON parsing failed
                json_response = llm_response["response"]
                extracted_data = None

            return {
                "answer": json_response,
                "metadata": {
                    "extraction_type": extraction_type or "general",
                    "extraction_successful": llm_response.get("is_valid_json", False),
                    "context_chunks": len(search_results),
                    "parsed_data": extracted_data,
                    "generation_time": llm_response.get("total_duration", 0),
                    "tokens_generated": llm_response.get("eval_count", 0)
                }
            }

        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {
                "answer": json.dumps({"error": f"Extraction failed: {str(e)}"}),
                "metadata": {"extraction_successful": False, "error": True}
            }
