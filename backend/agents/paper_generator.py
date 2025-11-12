import os
import json
from datetime import datetime
from markdown2 import markdown
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class PaperGeneratorAgent:
    def __init__(self):
        print("[PaperGenerator] 🚀 Initializing Paper Generator Agent...")
        self.llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
        self.results_dir = "backend/results/final_paper"
        os.makedirs(self.results_dir, exist_ok=True)

    # --------------------------------
    # Collect visualizations from multiple sources
    # --------------------------------
    def _collect_visualizations(self, experiment_results):
        """Collect all visualization paths from experiment results and results folders."""
        vis_paths = set()

        # 1. From experiment_results keys
        if "visualization_path" in experiment_results:
            vis_paths.add(experiment_results["visualization_path"])

        if "sample_visualization" in experiment_results:
            vis_paths.add(experiment_results["sample_visualization"])

        # 2. From dataset_analysis (Experiment Designer output)
        for ds in experiment_results.get("dataset_analysis", []):
            if "sample_visualization" in ds:
                vis_paths.add(ds["sample_visualization"])

        # 3. From backend/results/experiments directory
        exp_dir = "backend/results/experiments"
        if os.path.exists(exp_dir):
            for file in os.listdir(exp_dir):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    vis_paths.add(os.path.join(exp_dir, file))

        # 4. Validate and return
        vis_paths = [v for v in vis_paths if os.path.exists(v)]
        print(f"[PaperGenerator] 🖼️ Found {len(vis_paths)} visualizations to embed.")
        return vis_paths

    # --------------------------------
    # Compose Markdown draft
    # --------------------------------
    def _compose_markdown(self, domain_info, questions, data_info, experiment_results, critique):
        title = f"Autonomous Research on {domain_info.get('domain_name', 'Unknown Domain')}"
        abstract = (
            f"This paper explores the emerging field of {domain_info.get('domain_name', 'an identified domain')}, "
            f"autonomously investigated using AI-driven multi-agent research loops. "
            f"The system autonomously explores domains, generates hypotheses, finds datasets, "
            f"designs experiments, and critiques its outputs for refinement."
        )

        markdown_text = f"# {title}\n\n"
        markdown_text += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown_text += f"## Abstract\n{abstract}\n\n"
        markdown_text += f"## Introduction\n{domain_info.get('description', '')}\n\n"

        # Research Questions
        if questions:
            markdown_text += "## Research Questions\n"
            for i, q in enumerate(questions, 1):
                # Handle both string and dict questions
                question_text = q if isinstance(q, str) else q.get('question', str(q))
                markdown_text += f"**Q{i}:** {question_text}\n"
            markdown_text += "\n"

        # Dataset Summary
        markdown_text += "## Dataset Summary\n"
        for ds in data_info.get("metadata", []):
            out = ds.get("output", ds)  # Fallback to ds itself if no 'output' key
            # Handle different output structures
            if isinstance(out, dict):
                path = out.get('path', out.get('file_path', 'unknown'))
                dataset_type = out.get('type', 'tabular')
                files = out.get('files_inside', [])
            else:
                path = 'unknown'
                dataset_type = 'unknown'
                files = []
            
            markdown_text += f"- **Dataset:** {os.path.basename(path)}\n"
            markdown_text += f"  - Type: {dataset_type}\n"
            markdown_text += f"  - Files: {len(files)}\n\n"

        # Experiment Design
        markdown_text += "## Experiment Design and Methods\n"
        exp = experiment_results.get("experiment_proposal", {})
        if exp:
            markdown_text += "### Hypotheses\n"
            for h in exp.get("hypotheses", []):
                # Handle both dict and string hypotheses
                if isinstance(h, dict):
                    markdown_text += f"- **{h.get('hypothesis_id', '')}:** {h.get('description', '')}\n"
                    markdown_text += f"  - Predicted: {h.get('predicted_outcome', '')}\n"
                else:
                    markdown_text += f"- {str(h)}\n"
            markdown_text += "\n### Methods\n"
            for m in exp.get("experiment_design", {}).get("methods", []):
                # Handle both dict and string methods
                if isinstance(m, dict):
                    markdown_text += f"- {m.get('name', '')}: {m.get('description', '')}\n"
                else:
                    markdown_text += f"- {str(m)}\n"
            markdown_text += "\n"

        # Results and Visualizations
        markdown_text += "## Results and Visualizations\n"
        vis_paths = self._collect_visualizations(experiment_results)

        if vis_paths:
            for vis_path in vis_paths:
                caption = os.path.basename(vis_path).replace("_", " ").replace(".png", "").replace(".jpg", "")
                markdown_text += f"![{caption}]({vis_path})\n"
                markdown_text += f"*Figure: {caption.title()}*\n\n"
        else:
            markdown_text += "_No visualizations found._\n\n"

        # Critique Section
        markdown_text += "## Critique and Limitations\n"
        if critique and isinstance(critique, dict):
            markdown_text += f"**Critique Score:** {critique.get('critique_score', 'N/A')}\n\n"
            
            # Strengths
            strengths = critique.get("strengths", [])
            if strengths:
                markdown_text += "### Strengths\n"
                for s in strengths:
                    markdown_text += f"- {s}\n"
                markdown_text += "\n"
            
            # Weaknesses
            weaknesses = critique.get("weaknesses", [])
            if weaknesses:
                markdown_text += "### Weaknesses\n"
                for w in weaknesses:
                    markdown_text += f"- {w}\n"
                markdown_text += "\n"
            
            # Risks
            risks = critique.get("risks", [])
            if risks:
                markdown_text += "### Risks\n"
                for r in risks:
                    markdown_text += f"- {r}\n"
                markdown_text += "\n"
            
            # Recommended Fixes
            fixes = critique.get("recommended_fixes", [])
            if fixes:
                markdown_text += "### Recommended Fixes\n"
                for f in fixes:
                    if isinstance(f, dict):
                        fix_text = f.get('fix', '')
                        alt_text = f.get('alternative', '')
                        markdown_text += f"- {fix_text}"
                        if alt_text:
                            markdown_text += f" (Alternative: {alt_text})"
                        markdown_text += "\n"
                    else:
                        markdown_text += f"- {str(f)}\n"
                markdown_text += "\n"
        else:
            markdown_text += "_No critique available._\n\n"

        # Future Work
        markdown_text += "## Future Work\n"
        markdown_text += "Future iterations can include additional datasets, multi-modal reasoning, and reinforcement-based refinement of hypotheses.\n\n"

        markdown_text += "---\n*Generated autonomously by the Multi-Agent Research System.*\n"
        return markdown_text

    # --------------------------------
    # LLM Polishing
    # --------------------------------
    def polish_paper(self, markdown_text):
        print("[PaperGenerator] ✨ Polishing paper via LLM...")
        prompt = f"""
        You are a professional academic editor.
        Refine the following markdown-formatted research paper for clarity and style.
        Keep all image references and structure unchanged.
        Output Markdown only.

        {markdown_text}
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()

    # --------------------------------
    # Generate final paper (FIXED)
    # --------------------------------
    def generate_paper(self, domain_info, questions, data_info, experiment_results, critique):
        """
        Generate research paper with robust type handling
        
        All parameters can be either strings or dicts - this method normalizes them
        """
        print("[PaperGenerator] 🧠 Generating research paper...")
        
        try:
            # ========== NORMALIZE ALL INPUTS ==========
            
            # 1. Handle domain_info
            if isinstance(domain_info, str):
                domain_info = {"domain_name": domain_info, "description": ""}
            elif not isinstance(domain_info, dict):
                domain_info = {"domain_name": "Unknown Domain", "description": str(domain_info)}
            
            # 2. Handle questions
            if isinstance(questions, str):
                # Try parsing as JSON first
                try:
                    questions = json.loads(questions)
                except:
                    # If not JSON, treat as single question
                    questions = [questions]
            elif not isinstance(questions, list):
                questions = [str(questions)]
            
            # 3. Handle data_info
            if isinstance(data_info, str):
                try:
                    data_info = json.loads(data_info)
                except:
                    data_info = {"summary": data_info, "metadata": []}
            elif not isinstance(data_info, dict):
                data_info = {"summary": str(data_info), "metadata": []}
            
            # Ensure metadata key exists
            if "metadata" not in data_info:
                data_info["metadata"] = []
            
            # 4. Handle experiment_results (THIS WAS MISSING!)
            if isinstance(experiment_results, str):
                try:
                    experiment_results = json.loads(experiment_results)
                except:
                    experiment_results = {
                        "summary": experiment_results,
                        "experiment_proposal": {},
                        "dataset_analysis": []
                    }
            elif not isinstance(experiment_results, dict):
                experiment_results = {
                    "summary": str(experiment_results),
                    "experiment_proposal": {},
                    "dataset_analysis": []
                }
            
            # Ensure required keys exist
            if "experiment_proposal" not in experiment_results:
                experiment_results["experiment_proposal"] = {}
            if "dataset_analysis" not in experiment_results:
                experiment_results["dataset_analysis"] = []
            
            # 5. Handle critique
            if isinstance(critique, str):
                try:
                    critique = json.loads(critique)
                except:
                    critique = {
                        "notes": critique,
                        "strengths": [],
                        "weaknesses": [],
                        "risks": [],
                        "recommended_fixes": []
                    }
            elif not isinstance(critique, dict):
                critique = {"notes": str(critique)}
            
            # ========== GENERATE PAPER ==========
            raw_md = self._compose_markdown(domain_info, questions, data_info, experiment_results, critique)
            refined_md = self.polish_paper(raw_md)

            md_path = os.path.join(self.results_dir, "mini_research_paper.md")
            html_path = os.path.join(self.results_dir, "mini_research_paper.html")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(refined_md)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(markdown(refined_md))

            print(f"[PaperGenerator] ✅ Research paper saved:\n- {md_path}\n- {html_path}")
            
            return {
                "agent_name": "PaperGeneratorAgent",
                "status": "success",
                "markdown": md_path,
                "html": html_path
            }
        
        except Exception as e:
            print(f"[PaperGenerator] ❌ Error generating paper: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "agent_name": "PaperGeneratorAgent",
                "status": "error",
                "error": str(e)
            }


# -------------------------------
# Test run
# -------------------------------
if __name__ == "__main__":
    domain_info = {
        "domain_name": "AI for Scientific Discovery",
        "description": "AI is transforming how scientists explore patterns in molecular and experimental data."
    }

    questions = [
        "Can AI simulate experimental outcomes for unseen molecules?",
        "How can vision-based AI models enhance microscopic structure recognition?"
    ]

    data_info = {
        "metadata": [
            {"output": {"path": "backend/data/R111_S405_pH7_1p5pcPEG_noRNA.zip", "type": "non_tabular_zip", "files_inside": ["img1.tif", "img2.tif"]}}
        ]
    }

    experiment_results = {
        "dataset_analysis": [
            {"dataset_name": "R111_S405", "sample_visualization": "backend/results/experiments/R111_S405_pH7_1p5pcPEG_noRNA_samples.png"}
        ],
        "experiment_proposal": {
            "hypotheses": [
                {"hypothesis_id": "H1", "description": "RNA impacts image texture", "predicted_outcome": "Higher structure density"}
            ],
            "experiment_design": {"methods": [{"name": "Image Analysis", "description": "Analyze pixel intensity distributions"}]},
        },
        "visualization_path": "backend/results/experiments/structure_bar_chart.png"
    }

    critique = {
        "critique_score": 0.76,
        "strengths": ["Solid domain reasoning"],
        "weaknesses": ["Limited visual comparisons"],
        "risks": ["Overfitting on limited datasets"],
        "recommended_fixes": [{"fix": "Add more control experiments", "alternative": "Use simulated data for comparison"}]
    }

    agent = PaperGeneratorAgent()
    result = agent.generate_paper(domain_info, questions, data_info, experiment_results, critique)
    print(json.dumps(result, indent=2))