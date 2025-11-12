
"""
QuestionGeneratorAgent
----------------------
Generates 3–5 innovative, research-grade questions based on the
domain discovered by DomainScoutAgent.

Input: domain_info (dict)
Output: JSON list of questions with reasoning and impact fields

Dependencies:
    pip install langchain-groq langchain-core python-dotenv
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# Load environment variables from root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


class QuestionGeneratorAgent:
    """Formulates novel scientific research questions for the discovered domain."""

    def __init__(self):
        print("\n[QuestionGenerator] 🚀 Initializing research question generator...")
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.8,
            timeout=60,
        )

    # ---------------------------------------------------------------------
    def generate_questions(self, domain_info):
        """Generate structured research questions for the discovered domain."""
        if not domain_info or "domain_name" not in domain_info:
            print("[QuestionGenerator] ⚠️ No valid domain info provided.")
            return {"error": "Missing or invalid domain_info"}

        domain_name = domain_info.get("domain_name")
        domain_desc = domain_info.get("description", "No description available.")

        print(f"[QuestionGenerator] 🎯 Generating questions for: {domain_name}")

        template = PromptTemplate.from_template("""
        You are a senior research scientist tasked with proposing new, innovative questions
        in the domain of "{domain_name}".

        Domain context:
        {domain_desc}

        Requirements:
        - Generate 3 to 5 highly specific and novel research questions.
        - Each question must push scientific understanding forward after 2024.
        - Include reasoning behind why each question matters.
        - Include a brief potential impact if answered.

        Return a strict JSON array with fields:
        [
          {{
            "question": "...",
            "reasoning": "...",
            "potential_impact": "..."
          }}
        ]
        """)

        prompt = template.format(domain_name=domain_name, domain_desc=domain_desc)
        print("[QuestionGenerator] 🧠 Sending prompt to LLM...")

        start = time.time()
        try:
            response = self.llm.invoke(prompt)
            print(f"[QuestionGenerator] ✅ LLM responded in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"[QuestionGenerator] ❌ LLM call failed: {e}")
            return {"error": "LLM call failed", "details": str(e)}

        output_text = getattr(response, "content", str(response))
        parsed_data = self._extract_json(output_text)

        print(f"[QuestionGenerator] 🧾 Generated {len(parsed_data)} research questions.")
        return parsed_data

    # ---------------------------------------------------------------------
    def _extract_json(self, text: str):
        """Parse valid JSON list of questions, even if wrapped in markdown."""
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                data = [data]
            return data
        except Exception:
            import re
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return [{"error": "JSON parsing failed", "raw_output": text[:1000]}]


# ---------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------
# if __name__ == "__main__":
#     domain_info = {
#         "domain_name": "AI for Scientific Discovery",
#         "description": "The use of artificial intelligence to autonomously generate, validate, and optimize scientific hypotheses and experiments.",
#         "sources": ["https://www.weforum.org/publications/top-10-emerging-technologies-2024/"],
#         "confidence_score": 0.91
#     }

#     agent = QuestionGeneratorAgent()
#     questions = agent.generate_questions(domain_info)
#     print(json.dumps(questions, indent=2))
