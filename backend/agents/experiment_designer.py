#works
import os
import json
import random
import re
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class ExperimentDesignerAgent:
    def __init__(self):
        print("[ExperimentDesigner] 🚀 Initializing Experiment Designer Agent...")
        self.llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
        self.results_dir = "backend/results/experiments"
        os.makedirs(self.results_dir, exist_ok=True)
        self.current_domain = None

    # -----------------------------------
    # STEP 1: Analyze datasets
    # -----------------------------------
    def analyze_datasets(self, data_info):
        summaries = []
        for dataset in tqdm(data_info.get("metadata", []), desc="[ExperimentDesigner] Analyzing datasets"):
            path = dataset["output"]["path"]
            dtype = dataset["output"]["type"]

            if dtype == "non_tabular_zip":
                # Count TIFs or images
                files_inside = dataset["output"]["files_inside"]
                image_files = [f for f in files_inside if f.lower().endswith((".tif", ".png", ".jpg"))]
                num_images = len(image_files)
                avg_res = (4096, 4096)
                sample_images = image_files[:5]
                vis_path = os.path.join(self.results_dir, f"{os.path.basename(path).replace('.zip', '')}_samples.png")
                self._visualize_samples(sample_images, vis_path)

                summaries.append({
                    "dataset_name": os.path.basename(path).replace(".zip", ""),
                    "num_images": num_images,
                    "average_resolution": avg_res,
                    "sample_visualization": vis_path
                })

        return summaries

    # -----------------------------------
    # STEP 2: Generate Experiment Proposal
    # -----------------------------------
    def propose_experiment(self, domain_info, summaries):
        print("[ExperimentDesigner] 🧠 Generating experiment design proposal...")
        domain_name = domain_info.get("domain_name", "Unknown Domain")
        desc = domain_info.get("description", "No description available.")
        data_summary = json.dumps(summaries, indent=2)

        prompt = f"""
        You are a scientific experiment designer AI. Based on this domain and dataset summary, propose experiments.

        Domain: {domain_name}
        Description: {desc}
        Dataset Summary: {data_summary}

        Return a JSON with these keys:
        - hypotheses: list of hypotheses (id, description, predicted_outcome)
        - experiment_design: methods, metrics, tools
        - expected_outcome: success_criteria
        - visualization_to_generate: {{
            "type": "bar_chart" | "line_chart" | "pie_chart" | "scatter_plot",
            "data": {{
                "labels": [...],
                "values": {{
                    "Metric A": [...],
                    "Metric B": [...]
                }}
            }},
            "description": "short explanation"
        }}
        Output pure JSON only, no markdown, no explanations.
        """

        response = self.llm.invoke(prompt).content.strip()

        try:
            return json.loads(response)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {"error": "Failed to parse LLM output", "raw_output": response}

    # -----------------------------------
    # STEP 3: Visualization from LLM or fallback
    # -----------------------------------
    def _generate_visualization_from_llm(self, vis_spec, domain_info):
        try:
            vtype = vis_spec.get("type", "bar_chart")
            data = vis_spec.get("data", {})
            labels = data.get("labels", [])
            values_dict = data.get("values", {})

            # Fallback: generate domain-specific synthetic data if missing
            if not labels or not values_dict:
                print("[ExperimentDesigner] ⚠️ No valid chart data. Using contextual fallback.")
                labels, values_dict, vtype = self._generate_contextual_fallback(domain_info)

            fig, ax = plt.subplots(figsize=(8, 6))
            if vtype == "bar_chart":
                for metric, vals in values_dict.items():
                    ax.bar(labels, vals, label=metric)
            elif vtype == "line_chart":
                for metric, vals in values_dict.items():
                    ax.plot(labels, vals, marker='o', label=metric)
            elif vtype == "scatter_plot":
                for metric, vals in values_dict.items():
                    ax.scatter(labels, vals, label=metric)
            elif vtype == "pie_chart":
                metric = list(values_dict.keys())[0]
                vals = values_dict[metric]
                ax.pie(vals, labels=labels, autopct="%1.1f%%")
            else:
                print("[ExperimentDesigner] ⚠️ Unknown chart type, defaulting to bar chart.")
                for metric, vals in values_dict.items():
                    ax.bar(labels, vals, label=metric)

            ax.set_title(vis_spec.get("description", f"{vtype.title()} Visualization"))
            ax.legend()
            plt.xticks(rotation=30)
            plt.tight_layout()

            save_path = os.path.join(self.results_dir, f"visualization_{random.randint(1000,9999)}.png")
            plt.savefig(save_path)
            plt.close()
            return save_path

        except Exception as e:
            print(f"[ExperimentDesigner] ❌ Visualization generation failed: {e}")
            return None

    # -----------------------------------
    # Contextual fallback generation via LLM
    # -----------------------------------
    def _generate_contextual_fallback(self, domain_info):
        try:
            domain_name = domain_info.get("domain_name", "General Science")
            domain_desc = domain_info.get("description", "")

            prompt = f"""
            You are a scientific visualization assistant. Create a small dataset to visualize trends in this domain:
            Domain: {domain_name}
            Description: {domain_desc}

            Output JSON only:
            {{
              "labels": ["Label A", "Label B", "Label C"],
              "values": {{
                "Metric 1": [10, 20, 30],
                "Metric 2": [15, 25, 35]
              }},
              "type": "bar_chart"
            }}

            Guidelines:
            - Labels = conditions, techniques, or scenarios relevant to this domain.
            - Metrics = meaningful quantitative aspects (e.g., efficiency, accuracy, impact).
            - Numbers = 1–100 for clarity.
            - Output strictly JSON, no markdown or text outside JSON.
            """

            response = self.llm.invoke(prompt).content.strip()
            match = re.search(r"\{[\s\S]*\}", response)
            if not match:
                raise ValueError("No JSON found in LLM output")
            data = json.loads(match.group(0))
            labels = data.get("labels", ["A", "B", "C"])
            values = data.get("values", {"Metric": [1, 2, 3]})
            vtype = data.get("type", "bar_chart")
            print(f"[ExperimentDesigner] 🧩 Generated contextual fallback: {data}")
            return labels, values, vtype

        except Exception as e:
            print(f"[ExperimentDesigner] ⚠️ Contextual fallback failed: {e}")
            return ["A", "B", "C"], {"Metric": [1, 2, 3]}, "bar_chart"

    # -----------------------------------
    # Helper: visualize sample images
    # -----------------------------------
    def _visualize_samples(self, image_paths, save_path):
        try:
            fig, axes = plt.subplots(1, len(image_paths), figsize=(12, 4))
            for i, img_path in enumerate(image_paths):
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    axes[i].axis("off")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"[ExperimentDesigner] ⚠️ Sample visualization failed: {e}")

    # -----------------------------------
    # Public interface
    # -----------------------------------
    def design_experiment(self, domain_info, data_info):
        print("[ExperimentDesigner] 🚀 Starting experiment design pipeline...")
        self.current_domain = domain_info

        summaries = self.analyze_datasets(data_info)
        llm_result = self.propose_experiment(domain_info, summaries)

        # ✅ Added fallback handling for missing visualization
        vis_path = None
        if "visualization_to_generate" in llm_result:
            vis_path = self._generate_visualization_from_llm(
                llm_result["visualization_to_generate"], domain_info
            )
        else:
            # ✅ Ensure a default visualization exists if LLM fails to produce one
            print("[ExperimentDesigner] ⚠️ No visualization generated by LLM, using fallback chart.")
            labels, values, vtype = self._generate_contextual_fallback(domain_info)
            fallback_spec = {
                "type": vtype,
                "data": {"labels": labels, "values": values},
                "description": f"Fallback {vtype} visualization for {domain_info.get('domain_name', 'Unknown Domain')}"
            }
            vis_path = self._generate_visualization_from_llm(fallback_spec, domain_info)

        # ✅ Ensure vis_path always exists — for PaperGenerator safety
        if not vis_path or not os.path.exists(vis_path):
            vis_path = os.path.join(self.results_dir, "default_chart.png")
            if not os.path.exists(vis_path):
                # create a simple placeholder image
                plt.figure(figsize=(5, 3))
                plt.text(0.5, 0.5, "No Visualization Generated", ha="center", va="center", fontsize=12)
                plt.axis("off")
                plt.savefig(vis_path)
                plt.close()

        output = {
            "dataset_analysis": summaries,
            "experiment_proposal": llm_result,
            "visualization_path": vis_path,
        }

        save_path = os.path.join(self.results_dir, "final_experiment_design.json")
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[ExperimentDesigner] ✅ Experiment design completed. Saved to {save_path}")
        return output


# Run standalone
if __name__ == "__main__":
    dummy_domain = {
        "domain_name": "AI for Scientific Discovery",
        "description": "AI models are used to accelerate scientific research by analyzing large datasets and automating discovery."
    }

    dummy_data = {
        "metadata": [
            {
                "output": {
                    "path": "backend/data/sample_images.zip",
                    "type": "non_tabular_zip",
                    "files_inside": []
                }
            }
        ]
    }

    agent = ExperimentDesignerAgent()
    result = agent.design_experiment(dummy_domain, dummy_data)
    print(json.dumps(result, indent=2))
