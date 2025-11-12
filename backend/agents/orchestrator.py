import os
import json
from datetime import datetime
from dotenv import load_dotenv

# === Import all agents ===
from backend.agents.domain_scout import DomainScoutAgent
from backend.agents.question_generator import QuestionGeneratorAgent
from backend.agents.data_finder1 import DataFinderAgent
from backend.agents.experiment_designer import ExperimentDesignerAgent
from backend.agents.critic_agent import CriticAgent
from backend.agents.paper_generator import PaperGeneratorAgent
from backend.utils.vector_memory import MemoryManager

load_dotenv()

class Orchestrator:
    def __init__(self):
        print("\n🧩 Launching Autonomous Research Agentic System...\n")
        self.memory = MemoryManager()
        self.results_dir = "backend/results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize agents
        self.domain_agent = DomainScoutAgent()
        self.question_agent = QuestionGeneratorAgent()
        self.data_agent = DataFinderAgent()
        self.experiment_agent = ExperimentDesignerAgent()
        self.critic_agent = CriticAgent()
        self.paper_agent = PaperGeneratorAgent()

    def run_cycle(self, iterations=1):
        """Run the full autonomous research cycle."""
        run_summary = {"results": []}

        for i in range(iterations):
            print(f"\n🧠 Iteration {i+1}/{iterations}\n")

            try:
                result = self._run_once()
                run_summary["results"].append({
                    "iteration": i + 1,
                    "status": "success",
                    **result
                })
            except Exception as e:
                print(f"[Orchestrator] ❌ Iteration {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
                run_summary["results"].append({
                    "iteration": i + 1,
                    "status": "failed",
                    "error": str(e)
                })

        run_summary["status"] = "success"
        run_summary["completed_at"] = datetime.now().isoformat()

        with open(os.path.join(self.results_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

        print("\n✅ Research pipeline completed successfully!")
        print(f"📁 Results saved at: {os.path.join(self.results_dir, 'run_summary.json')}")
        return run_summary

    def _run_once(self):
        summary = {}

        # === STEP 1: DOMAIN DISCOVERY ===
        print("[Orchestrator] 🌐 Launching DomainScoutAgent...")
        domain_info = self.domain_agent.discover_domain()
        
        # Store structured data + text summary
        self.memory.add("domain", domain_info)  # ← NEW: Store the actual dict
        self.memory.add_summary(domain_info["description"], {"stage": "domain"})
        
        summary["domain"] = domain_info

        # === STEP 2: QUESTION GENERATION ===
        print("\n[Orchestrator] 💭 Launching QuestionGeneratorAgent...")
        questions = self.question_agent.generate_questions(domain_info)
        
        # Store structured data + text summary
        self.memory.add("questions", questions)  # ← NEW: Store the actual list/dict
        self.memory.add_summary(str(questions), {"stage": "questions"})
        
        summary["questions"] = questions

        # === STEP 3: DATA FINDING ===
        print("\n[Orchestrator] 🧱 Launching DataFinderAgent...")
        try:
            data_info = self.data_agent.find_data(domain_info["domain_name"], questions)
        except Exception as e:
            print(f"[Orchestrator] ⚠️ DataFinder failed: {e}")
            data_info = {"metadata": [], "summary": "DataFinder failed."}
        
        # Store structured data + text summary
        self.memory.add("data", data_info)  # ← NEW: Store the actual dict
        self.memory.add_summary(data_info.get("summary", "Data discovery complete"), {"stage": "data"})
        
        summary["data_info"] = data_info

        # === STEP 4: EXPERIMENT DESIGN ===
        print("\n[Orchestrator] ⚗️ Launching ExperimentDesignerAgent...")
        try:
            experiment_results = self.experiment_agent.design_experiment(domain_info, data_info)
        except Exception as e:
            print(f"[Orchestrator] ⚠️ Experiment design failed: {e}")
            experiment_results = {"error": str(e)}
        
        # Store structured data + text summary
        self.memory.add("experiment", experiment_results)  # ← NEW: Store the actual dict
        self.memory.add_summary("Experiment design completed.", {"stage": "experiment"})
        
        summary["experiment_results"] = experiment_results

        # === STEP 5: CRITIC AGENT ===
        print("\n[Orchestrator] 🧠 Launching CriticAgent...")
        try:
            critique = self.critic_agent.critique(domain_info, experiment_results)
        except Exception as e:
            print(f"[Orchestrator] ⚠️ CriticAgent failed: {e}")
            critique = {"error": str(e)}
        
        # Store structured data + text summary
        self.memory.add("critique", critique)  # ← NEW: Store the actual dict
        self.memory.add_summary("Critique completed.", {"stage": "critique"})
        
        summary["critique"] = critique

        # === STEP 6: PAPER GENERATION ===
        print("\n[Orchestrator] 📄 Launching PaperGeneratorAgent...")
        try:
            # Retrieve structured data from memory (not strings!)
            paper = self.paper_agent.generate_paper(
                domain_info=self.memory.get("domain"),      # ← Get dict from memory
                questions=self.memory.get("questions"),      # ← Get list from memory
                data_info=self.memory.get("data"),          # ← Get dict from memory
                experiment_results=self.memory.get("experiment"),  # ← Get dict from memory
                critique=self.memory.get("critique")        # ← Get dict from memory
            )
        except Exception as e:
            print(f"[Orchestrator] ⚠️ Paper generation failed: {e}")
            import traceback
            traceback.print_exc()
            paper = {"error": str(e)}
        
        # Store structured data + text summary
        self.memory.add("paper", paper)  # ← NEW: Store the actual dict
        self.memory.add_summary("Paper generation completed.", {"stage": "paper"})
        
        summary["paper"] = paper

        return summary


# === ENTRY POINT ===
if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run_cycle(iterations=1)  # Start with 1 iteration for testing