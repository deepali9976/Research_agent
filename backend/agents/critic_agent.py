"""
CriticAgent
-----------
Evaluates experiment proposals and results produced by ExperimentDesignerAgent.

Outputs a structured JSON with:
- strengths
- weaknesses
- risks
- recommended fixes / next steps
- numerical critique_score (0-1)
- iterate (True/False) + suggested changes for next iteration

Behavior:
- Primary: Use remote LLM (ChatGroq) for nuanced critique.
- Fallback: Use simple rule-based heuristics if LLM not available.
"""

import os
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv

# LLM client — uses same API wrapper as other agents
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
except Exception:
    ChatGroq = None
    PromptTemplate = None

load_dotenv()


class CriticAgent:
    def __init__(self, groq_model: str = "llama-3.1-8b-instant"):
        self.console_prefix = "[CriticAgent]"
        self.llm = None
        if ChatGroq and os.getenv("GROQ_API_KEY"):
            try:
                self.llm = ChatGroq(model=groq_model, api_key=os.getenv("GROQ_API_KEY"))
                print(f"{self.console_prefix} ✅ LLM initialized (Groq).")
            except Exception as e:
                print(f"{self.console_prefix} ⚠️ Failed to init LLM: {e}")
                self.llm = None
        else:
            if not ChatGroq:
                print(f"{self.console_prefix} ⚠️ ChatGroq client not installed.")
            else:
                print(f"{self.console_prefix} ⚠️ GROQ_API_KEY not found. Using rule-based fallback.")

    # ------------------------------------------------------------------
    def _llm_critique(self, domain_info: dict, experiment_result: dict):
        """
        Ask the LLM to critique the experiment and propose improvements.
        Returns parsed JSON or raises if LLM fails.
        """

        if not self.llm or not PromptTemplate:
            raise RuntimeError("LLM not available")

        template = PromptTemplate.from_template("""
        You are a critical, constructive peer-reviewer for computational experiments.
        Given the research domain, dataset analysis and the proposed experiment plan, 
        produce a structured critique.

        Domain:
        {domain_info}

        Dataset analysis:
        {dataset_analysis}

        Experiment proposal:
        {experiment_proposal}

        Provide a JSON object with the following fields:
        - strengths: list of concise strengths (max 6)
        - weaknesses: list of concise weaknesses (max 6)
        - risks: list of possible risks or invalid assumptions (max 6)
        - recommended_fixes: array of short actionable fixes or alternative approaches
        - critique_score: number between 0 and 1 (higher = better)
        - iterate: boolean (should we iterate another cycle?)
        - suggested_next_steps: list of 3 prioritized steps (short text)
        - notes: any additional brief comments

        Be concise. Output only valid JSON.
        """)

        prompt = template.format(
            domain_info=json.dumps(domain_info, indent=2),
            dataset_analysis=json.dumps(experiment_result.get("dataset_analysis", []), indent=2),
            experiment_proposal=json.dumps(experiment_result.get("experiment_proposal", {}), indent=2)
        )

        print(f"{self.console_prefix} 🧠 Sending critique prompt to LLM...")
        response = self.llm.invoke(prompt)
        # Attempt safe JSON extraction
        try:
            parsed = json.loads(response.content)
            return parsed
        except Exception:
            import re
            m = re.search(r"\{.*\}", response.content, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise

    # ------------------------------------------------------------------
    def _heuristic_critique(self, domain_info: dict, experiment_result: dict):
        """
        Lightweight rule-based fallback critique when LLM is unavailable.
        Produces the same JSON structure but simpler.
        """
        strengths = []
        weaknesses = []
        risks = []
        fixes = []
        suggested_next_steps = []

        # Basic checks
        ds = experiment_result.get("dataset_analysis", [])
        proposal = experiment_result.get("experiment_proposal", {})

        # Strengths heuristic
        if ds:
            strengths.append("Datasets are real scientific data and contain images suitable for image analysis.")
            strengths.append(f"Found {len(ds)} dataset(s) with sample visualizations.")
        else:
            weaknesses.append("No datasets available for analysis.")

        # Check methods present
        methods = proposal.get("experiment_design", {}).get("methods", []) if isinstance(proposal, dict) else []
        if methods:
            strengths.append("Experiment proposal includes concrete methods.")
        else:
            weaknesses.append("Experiment proposal lacks concrete methods.")

        # Check metrics
        metrics = proposal.get("experiment_design", {}).get("metrics", []) if isinstance(proposal, dict) else []
        if metrics:
            strengths.append("Relevant metrics have been proposed.")
        else:
            weaknesses.append("No clear evaluation metrics provided.")

        # Risks
        if any(d.get("num_images", 0) < 10 for d in ds):
            risks.append("Small sample size in at least one dataset may limit statistical power.")
        if not metrics:
            risks.append("No quantitative evaluation may lead to ambiguous conclusions.")
        # Tool realism
        tools = proposal.get("experiment_design", {}).get("tools", [])
        if any(t.get("name", "").lower() in ("tensorflow", "pytorch") for t in tools):
            strengths.append("Proposal references robust ML toolkits for model training.")
        else:
            weaknesses.append("No mainstream ML toolset specified (TensorFlow/PyTorch).")

        # Recommended fixes
        if not metrics:
            fixes.append("Define 2-3 measurable quantitative metrics with thresholds for success.")
        if any(d.get("num_images", 0) < 10 for d in ds):
            fixes.append("Aggregate smaller datasets or use data augmentation to increase sample size.")
        fixes.append("Add a control group or baseline model to quantify improvement.")

        # Suggested next steps (prioritized)
        suggested_next_steps = [
            "Run EDA to compute class imbalance and per-condition sample counts.",
            "Implement a simple baseline model (e.g., logistic regression on handcrafted features) to get baseline metrics.",
            "If image counts are low, apply augmentation or combine similar datasets."
        ]

        # score heuristic: base 0.5 + bonuses/penalties
        score = 0.5
        score += 0.15 * (1 if methods else -0.1)
        score += 0.1 * (1 if metrics else -0.15)
        score += 0.1 * (0 if risks else 0.05)
        score = max(0.0, min(1.0, score))

        iterate = bool(weaknesses or risks or score < 0.8)

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "risks": risks,
            "recommended_fixes": fixes,
            "critique_score": round(score, 3),
            "iterate": iterate,
            "suggested_next_steps": suggested_next_steps,
            "notes": "Used heuristic fallback critique (LLM not available or failed)."
        }

    # ------------------------------------------------------------------
    def critique(self, domain_info: dict, experiment_result: dict):
        """
        Public API:
        - domain_info: dict from DomainScoutAgent
        - experiment_result: dict from ExperimentDesignerAgent
        Returns structured critique JSON.
        """
        try:
            if self.llm:
                try:
                    result = self._llm_critique(domain_info, experiment_result)
                    # validate minimal fields
                    for k in ("strengths", "weaknesses", "recommended_fixes", "critique_score", "iterate"):
                        if k not in result:
                            raise ValueError(f"Missing key from LLM result: {k}")
                    print(f"{self.console_prefix} ✅ LLM critique succeeded.")
                    return result
                except Exception as e:
                    print(f"{self.console_prefix} ⚠️ LLM critique failed: {e}")
                    traceback.print_exc()
                    print(f"{self.console_prefix} 🔁 Falling back to heuristic critique.")
                    return self._heuristic_critique(domain_info, experiment_result)
            else:
                return self._heuristic_critique(domain_info, experiment_result)
        except Exception as e:
            print(f"{self.console_prefix} ❌ Unexpected error in critique: {e}")
            traceback.print_exc()
            # Safe fallback minimal response
            return {
                "strengths": [],
                "weaknesses": ["Critic agent failed unexpectedly."],
                "risks": [],
                "recommended_fixes": ["Inspect logs."],
                "critique_score": 0.0,
                "iterate": True,
                "suggested_next_steps": ["Check CriticAgent logs for exceptions."],
                "notes": f"Exception: {str(e)}"
            }


# ------------------------------------------------------------------
# Standalone test runner
# ------------------------------------------------------------------
# if __name__ == "__main__":
#     # quick smoke test with mock input
#     from datetime import datetime

#     domain_info = {
#         "domain_name": "AI for Scientific Discovery",
#         "description": "Use ML to accelerate discovery",
#         "sources": []
#     }

#     # minimal experiment_result shape (example)
#     experiment_result = {
#         "dataset_analysis": [
#             {"dataset_name": "demo", "num_images": 5}
#         ],
#         "experiment_proposal": {
#             "hypotheses": [{"hypothesis_id": "H1", "description": "Demo"}],
#             "experiment_design": {
#                 "methods": [{"name": "Image Analysis", "description": "Use CV"}],
#                 "metrics": [{"name": "Accuracy", "description": "Binary accuracy"}],
#                 "tools": [{"name": "TensorFlow", "description": "train models"}]
#             },
#             "visualization_to_generate": {}
#         }
#     }

#     agent = CriticAgent()
#     critique = agent.critique(domain_info, experiment_result)
#     print(json.dumps(critique, indent=2))
