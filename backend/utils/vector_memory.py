"""
Vector Memory Manager (Fixed - Handles Both Structured + Vector Data)
----------------------------------------------------------------------
Stores structured agent outputs AND text embeddings.
No torch, no internet, no HuggingFace token required.
"""

import json
import traceback
from pathlib import Path
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS, Chroma


class MemoryManager:
    def __init__(self, storage_path: str = "backend/memory_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # NEW: Store structured data separately
        self.structured_memory = {}
        self.memory_file = self.storage_path / "structured_memory.json"
        
        # Original vector store
        self.vectorstore = None

        print("[MemoryManager] 🚀 Initializing OFFLINE vector memory (FakeEmbeddings)...")

        try:
            self.embedding_model = FakeEmbeddings(size=384)  # CPU-safe mock embeddings
            self.vectorstore = FAISS.from_texts(["Memory initialized."], embedding=self.embedding_model)
            print("[MemoryManager] ✅ FAISS initialized successfully (offline).")
        except Exception as e:
            print(f"[MemoryManager] ⚠️ FAISS failed ({e}). Falling back to Chroma.")
            try:
                self.vectorstore = Chroma(embedding_function=self.embedding_model)
                print("[MemoryManager] ✅ Using Chroma fallback.")
            except Exception as e2:
                print(f"[MemoryManager] ❌ Memory system disabled: {e2}")
                self.vectorstore = None
        
        # Load existing structured memory
        self._load_structured()

    # ==================== NEW: STRUCTURED MEMORY ====================
    
    def add(self, key: str, value):
        """
        Add structured data (dicts, lists) to memory.
        Use this for storing agent outputs.
        
        Example:
            memory.add("questions", ["Q1", "Q2"])
            memory.add("data", {"datasets": [...]})
        """
        self.structured_memory[key] = value
        self._save_structured()
        print(f"[MemoryManager] 💾 Added memory: {key}")
    
    def get(self, key: str, default=None):
        """
        Retrieve structured data from memory.
        
        Returns the actual value (dict/list), not a string!
        """
        return self.structured_memory.get(key, default)
    
    def get_all(self):
        """Get all structured memories"""
        return self.structured_memory
    
    def _save_structured(self):
        """Save structured memory to JSON"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.structured_memory, f, indent=2, default=str)
        except Exception as e:
            print(f"[MemoryManager] ⚠️ Save failed: {e}")
    
    def _load_structured(self):
        """Load structured memory from JSON"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    self.structured_memory = json.load(f)
                print(f"[MemoryManager] 📂 Loaded {len(self.structured_memory)} memories")
        except Exception as e:
            print(f"[MemoryManager] ⚠️ Load failed: {e}")

    # ==================== ORIGINAL: VECTOR MEMORY ====================
    
    def add_summary(self, text: str, metadata: dict = None):
        """
        Add text summary to vector store for semantic search.
        
        Example:
            memory.add_summary("Found 3 datasets about climate", {"stage": "data"})
        """
        if not self.vectorstore:
            print("[MemoryManager] ⚠️ No active vector store.")
            return
        try:
            metadata = metadata or {}
            self.vectorstore.add_texts([text], metadatas=[metadata])
            print(f"[MemoryManager] 💾 Added summary: {metadata.get('stage', 'unknown')}")
        except Exception as e:
            print(f"[MemoryManager] ❌ Failed to add summary: {e}")
            traceback.print_exc()

    def query(self, query_text: str, k: int = 3):
        """
        Semantic search across text summaries.
        
        Example:
            memories = memory.query("what datasets did we find?", k=3)
        """
        if not self.vectorstore:
            print("[MemoryManager] ⚠️ No vectorstore available.")
            return []
        try:
            docs = self.vectorstore.similarity_search(query_text, k=k)
            print(f"[MemoryManager] 🔍 Retrieved {len(docs)} memories.")
            return [d.page_content for d in docs]
        except Exception as e:
            print(f"[MemoryManager] ❌ Retrieval failed: {e}")
            return []

    def save(self):
        """Persist FAISS index to disk."""
        if not self.vectorstore or not isinstance(self.vectorstore, FAISS):
            return
        try:
            save_dir = self.storage_path / "faiss_index"
            self.vectorstore.save_local(str(save_dir))
            print(f"[MemoryManager] 💽 Saved FAISS index at: {save_dir}")
        except Exception as e:
            print(f"[MemoryManager] ⚠️ Save failed: {e}")

    def load(self):
        """Reload FAISS index."""
        try:
            index_path = self.storage_path / "faiss_index"
            if index_path.exists():
                self.vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print(f"[MemoryManager] 🔄 Loaded FAISS index from: {index_path}")
        except Exception as e:
            print(f"[MemoryManager] ⚠️ Load failed: {e}")


# ==================== TEST ====================
if __name__ == "__main__":
    memory = MemoryManager()
    
    # Test 1: Store structured data (for agents)
    memory.add("questions", ["What is AI?", "How does it work?"])
    memory.add("data", {"datasets": [{"path": "data.csv", "rows": 100}]})
    
    # Test 2: Store text summaries (for semantic search)
    memory.add_summary("Found 3 climate datasets with 1000 rows", {"stage": "data"})
    memory.add_summary("Designed regression experiment", {"stage": "experiment"})
    
    # Test 3: Retrieve structured data
    print("\n[TEST] Structured Memory:")
    questions = memory.get("questions")
    print(f"Type: {type(questions)}")  # Should be list
    print(f"Questions: {questions}")
    
    data = memory.get("data")
    print(f"Type: {type(data)}")  # Should be dict
    print(f"Data: {data}")
    
    # Test 4: Semantic search
    print("\n[TEST] Vector Search:")
    results = memory.query("what datasets?", k=2)
    print(f"Found: {results}")
    
    # Save everything
    memory.save()
    print("\n[TEST] Memory saved to:", memory.storage_path)