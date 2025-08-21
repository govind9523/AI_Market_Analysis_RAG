"""
Autonomous tool routing system - Bonus Feature
Automatically determines which tool to use based on query analysis
"""

from typing import Tuple, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

class AutonomousRouter:
    """Autonomous router for determining which RAG tool to use"""

    def __init__(self, llm_model):
        """
        Initialize autonomous router

        Args:
            llm_model: LLM for intelligent routing decisions
        """
        self.llm_model = llm_model

        # Rule-based patterns for fast routing (fallback for LLM)
        self.patterns = {
            "qa": [
                r"what (is|are|was|were|does|do)",
                r"how (much|many|often|long)",
                r"which|where|when|who|why",
                r"\?",  # Questions usually have question marks
                r"(explain|tell me about|describe)",
                r"market share of",
                r"percentage|percent"
            ],
            "summarize": [
                r"summarize|summary",
                r"overview|brief",
                r"key points|main points",
                r"in summary|overall",
                r"give me (a|the) (summary|overview)",
                r"what are the (main|key|important)"
            ],
            "extract": [
                r"extract|extraction",
                r"(get|find|list) (the )?(data|information|numbers|figures)",
                r"json|structured data",
                r"financial (data|information|metrics)",
                r"key (data|metrics|numbers|figures)",
                r"(market|revenue|growth) (data|numbers)",
                r"SWOT (data|analysis)"
            ]
        }

    def _rule_based_routing(self, query: str) -> Tuple[str, float]:
        """
        Fast rule-based routing using patterns

        Args:
            query: User query

        Returns:
            Tuple of (tool_name, confidence_score)
        """
        query_lower = query.lower()
        scores = {"qa": 0, "summarize": 0, "extract": 0}

        # Score each tool based on pattern matches
        for tool, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[tool] += 1

        # Determine winner
        if all(score == 0 for score in scores.values()):
            # No patterns matched, default to Q&A
            return "qa", 0.3

        best_tool = max(scores, key=scores.get)
        max_score = scores[best_tool]

        # Calculate confidence (simple heuristic)
        total_matches = sum(scores.values())
        confidence = max_score / max(total_matches, 1)

        return best_tool, min(confidence, 0.95)  # Cap confidence

    async def _llm_based_routing(self, query: str) -> Tuple[str, float]:
        """
        LLM-based routing for complex queries

        Args:
            query: User query

        Returns:
            Tuple of (tool_name, confidence_score)
        """
        try:
            system_prompt = """You are a query classifier for a RAG system analyzing a market research report. 

Your task: Classify queries into one of three categories:
1. "qa" - Specific questions about the report content (who, what, when, where, why, how questions)
2. "summarize" - Requests for summaries, overviews, or general insights
3. "extract" - Requests for structured data, specific numbers, JSON format, or data extraction

Respond with only the category name (qa, summarize, or extract) and a confidence score (0-1).
Format: <category>|<confidence>

Examples:
- "What is FutureFlow's market share?" → qa|0.9
- "Summarize the key findings" → summarize|0.95
- "Extract financial data as JSON" → extract|0.9
- "Give me an overview" → summarize|0.8"""

            user_prompt = f"Classify this query: '{query}'"

            response = await self.llm_model.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "num_predict": 20    # Short response expected
                }
            )

            # Parse response
            response_text = response["response"].strip()
            if "|" in response_text:
                tool, conf_str = response_text.split("|", 1)
                tool = tool.strip().lower()
                confidence = float(conf_str.strip())

                if tool in ["qa", "summarize", "extract"] and 0 <= confidence <= 1:
                    return tool, confidence

            # Fallback if parsing failed
            logger.warning(f"LLM routing parse failed: {response_text}")
            return self._rule_based_routing(query)

        except Exception as e:
            logger.warning(f"LLM routing failed: {e}, falling back to rules")
            return self._rule_based_routing(query)

    async def route_query(self, query: str, use_llm: bool = True) -> Tuple[str, float]:
        """
        Route query to appropriate tool

        Args:
            query: User query to route
            use_llm: Whether to use LLM for routing (vs rule-based only)

        Returns:
            Tuple of (tool_name, confidence_score)
        """
        try:
            logger.info(f"Routing query: {query[:50]}...")

            # First try rule-based routing for speed
            rule_tool, rule_confidence = self._rule_based_routing(query)

            # If rule-based is very confident, use it
            if rule_confidence >= 0.8:
                logger.info(f"Rule-based routing: {rule_tool} (confidence: {rule_confidence})")
                return rule_tool, rule_confidence

            # Otherwise, use LLM for better accuracy (if enabled)
            if use_llm:
                llm_tool, llm_confidence = await self._llm_based_routing(query)
                logger.info(f"LLM-based routing: {llm_tool} (confidence: {llm_confidence})")
                return llm_tool, llm_confidence
            else:
                logger.info(f"Rule-based routing: {rule_tool} (confidence: {rule_confidence})")
                return rule_tool, rule_confidence

        except Exception as e:
            logger.error(f"Routing failed: {e}, defaulting to Q&A")
            return "qa", 0.3

    def get_routing_explanation(self, query: str, selected_tool: str, confidence: float) -> Dict[str, Any]:
        """Get explanation of routing decision"""
        rule_tool, rule_confidence = self._rule_based_routing(query)

        return {
            "query": query,
            "selected_tool": selected_tool,
            "confidence": confidence,
            "rule_based_suggestion": rule_tool,
            "rule_based_confidence": rule_confidence,
            "routing_method": "llm" if confidence != rule_confidence else "rules",
            "available_tools": {
                "qa": "Answer specific questions about the report",
                "summarize": "Provide summaries and overviews",
                "extract": "Extract structured data in JSON format"
            }
        }
