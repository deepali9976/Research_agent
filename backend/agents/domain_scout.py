# backend/agents/domain_scout.py
import json
import os
import re
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

class DomainScoutAgent:
    def __init__(self):
        print("[DomainScout] 🚀 Initializing autonomous domain scout agent...")
        self.search = TavilySearch(max_results=5)
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def _extract_json(self, text: str):
        """Extract valid JSON from model output that may include markdown or extra text."""
        cleaned = text.replace("```json", "").replace("```", "").strip()

        # Try direct parse first
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                parsed = sorted(parsed, key=lambda x: x.get("confidence_score", 0), reverse=True)[0]
            return parsed
        except json.JSONDecodeError:
            pass

        # Regex fallback
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        # Graceful failure
        return {"error": "JSON parsing failed", "raw_output": text[:800]}

    def discover_domain(self):
        """Let the LLM autonomously decide which domain type is most relevant post-2024."""
        print("[DomainScout] 🔍 Searching for emerging domains across all fields...")

        query = (
            "emerging research or technology domains after January 2024 "
            "(include scientific, technological, environmental, and interdisciplinary trends)"
        )

        search_results = self.search.run(query)

        template = PromptTemplate.from_template("""
        You are an autonomous scientific domain scout.

        Given these recent search results:
        {search_results}

        Step 1: Analyze all fields — scientific, technological, social, environmental, interdisciplinary.
        Step 2: Identify 5 new or emerging domains that became relevant after 2024.
        Step 3: Pick ONE domain that seems most promising for research based on:
        - novelty,
        - potential impact,
        - available data or research opportunities.

        Return **only** a valid JSON with fields:
        {{
            "domain_name": "string",
            "description": "string",
            "sources": ["list of URLs"],
            "confidence_score": float (0–1)
        }}
        """)

        prompt = template.format(search_results=search_results)
        response = self.llm.invoke(prompt)

        parsed = self._extract_json(response.content)
        print(f"[DomainScout] ✅ Domain identified: {parsed.get('domain_name', 'Unknown')}")
        return parsed


# Run directly
if __name__ == "__main__":
    agent = DomainScoutAgent()
    result = agent.discover_domain()
    print(json.dumps(result, indent=2))
