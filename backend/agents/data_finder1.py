"""
DataFinderAgent - Simple & Clean Version for Multi-Agent System
================================================================
Purpose: Automatically finds and downloads datasets for emerging research domains

Key Features:
1. Multi-source search (DuckDuckGo, Data.gov API, Zenodo)
2. Smart filtering (only downloads actual dataset files)
3. Basic data cleaning (CSV/JSON/Excel)
4. Returns structured output for orchestrator
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse


class DataFinderAgent:
    """
    Agent that discovers and downloads datasets for scientific research domains.
    Works as part of a larger multi-agent orchestration system.
    """
    
    def __init__(self, output_dir="backend/data", max_datasets=3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_datasets = max_datasets
        print(f"[DataFinder] ✅ Agent initialized")

    # ==================== PUBLIC METHOD ====================
    def find_data(self, domain_info, questions):
        """
        Main method called by orchestrator.
        
        Args:
            domain_info: Dict with 'domain_name' or string
            questions: List of research questions (for future use)
            
        Returns:
            Dict with agent results for orchestrator
        """
        # Extract domain name
        domain = domain_info.get("domain_name") if isinstance(domain_info, dict) else str(domain_info)
        print(f"\n[DataFinder] 🔍 Searching for datasets: '{domain}'")
        
        # Step 1: Search for dataset URLs
        urls = self._search_multiple_sources(domain)
        
        # Step 2: Download and process datasets
        datasets = self._process_datasets(urls, domain)
        
        # Step 3: Return structured output for orchestrator
        return {
            "agent_name": "DataFinderAgent",
            "status": "success" if datasets else "no_data_found",
            "domain": domain,
            "datasets_found": len(datasets),
            "datasets": datasets,
            "summary": f"Found {len(datasets)} valid datasets for '{domain}'"
        }

    # ==================== SEARCH METHODS ====================
    def _search_multiple_sources(self, domain):
        """Search multiple data sources and combine results"""
        all_urls = []
        
        # Source 1: Data.gov API (US Government datasets)
        all_urls.extend(self._search_datagov(domain))
        
        # Source 2: Zenodo (Scientific research datasets)
        all_urls.extend(self._search_zenodo(domain))
        
        # Remove duplicates and limit results
        unique_urls = list(set(all_urls))[:self.max_datasets * 2]
        print(f"[DataFinder] 📊 Found {len(unique_urls)} candidate URLs")
        return unique_urls

    def _search_datagov(self, domain):
        """Search Data.gov using their official API"""
        try:
            url = "https://catalog.data.gov/api/3/action/package_search"
            params = {"q": domain, "rows": 5}
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            dataset_urls = []
            for package in data.get("result", {}).get("results", []):
                for resource in package.get("resources", []):
                    file_url = resource.get("url", "")
                    if self._is_dataset_file(file_url):
                        dataset_urls.append(file_url)
            
            print(f"[DataFinder]   → Data.gov: {len(dataset_urls)} datasets")
            return dataset_urls
        except Exception as e:
            print(f"[DataFinder]   → Data.gov failed: {e}")
            return []

    def _search_zenodo(self, domain):
        """Search Zenodo (open science repository) using their API"""
        try:
            url = "https://zenodo.org/api/records"
            params = {"q": domain, "type": "dataset", "size": 5}
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            dataset_urls = []
            for record in data.get("hits", {}).get("hits", []):
                for file in record.get("files", []):
                    file_url = file.get("links", {}).get("self")
                    if file_url and self._is_dataset_file(file_url):
                        dataset_urls.append(file_url)
            
            print(f"[DataFinder]   → Zenodo: {len(dataset_urls)} datasets")
            return dataset_urls
        except Exception as e:
            print(f"[DataFinder]   → Zenodo failed: {e}")
            return []

    # ==================== FILTERING & VALIDATION ====================
    def _is_dataset_file(self, url):
        """Check if URL points to an actual dataset file (not a webpage)"""
        if not url or not url.startswith("http"):
            return False
        
        # Exclude search/browse pages
        if any(x in url.lower() for x in ["/search", "/browse", "/explore"]):
            return False
        
        # Must be a data file
        return any(url.lower().endswith(ext) for ext in [".csv", ".json", ".xlsx", ".xls"])

    # ==================== DOWNLOAD & PROCESSING ====================
    def _process_datasets(self, urls, domain):
        """Download and clean datasets, return metadata"""
        datasets = []
        
        for url in urls[:self.max_datasets]:
            result = self._download_and_clean(url, domain)
            if result:
                datasets.append(result)
        
        print(f"[DataFinder] ✅ Successfully processed {len(datasets)} datasets")
        return datasets

    def _download_and_clean(self, url, domain):
        """Download single dataset and perform basic cleaning"""
        try:
            # Get filename
            filename = os.path.basename(urlparse(url).path) or "dataset.csv"
            filepath = self.output_dir / filename
            
            print(f"[DataFinder] ⬇️  Downloading: {filename}")
            
            # Download file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            # Load into pandas
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif filename.endswith(".json"):
                df = pd.read_json(filepath)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(filepath)
            else:
                return None
            
            # Basic cleaning
            df = df.dropna(how="all").drop_duplicates()
            
            # Save cleaned version
            cleaned_path = self.output_dir / f"cleaned_{filename}"
            df.to_csv(cleaned_path, index=False)
            
            # Return metadata for orchestrator
            return {
                "source_url": url,
                "file_path": str(cleaned_path),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "size_mb": round(filepath.stat().st_size / (1024**2), 2)
            }
            
        except Exception as e:
            print(f"[DataFinder] ❌ Failed to process {url}: {e}")
            return None


# ==================== TEST CODE ====================
if __name__ == "__main__":
    # Example usage
    agent = DataFinderAgent()
    
    domain = {"domain_name": "climate change"}
    questions = ["What are CO2 emission trends?"]
    
    results = agent.find_data(domain, questions)
    print("\n" + "="*60)
    print(json.dumps(results, indent=2))