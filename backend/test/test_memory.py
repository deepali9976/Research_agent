# backend/tests/test_memory.py
#works
from backend.utils.vector_memory import MemoryManager

def test_memory():
    mem = MemoryManager(persist_dir="backend/memory/test_chroma", summary_log="backend/memory/test_summaries.json")
    mem.clear_memory()
    mem.add_memory("domain_scout", "Found domain: Quantum Neuromorphic Materials", {"domain": "Quantum Neuromorphic Materials"})
    mem.add_summary("Selected Quantum Neuromorphic Materials as candidate domain.", {"domain": "Quantum Neuromorphic Materials"})
    ctx = mem.get_context_for_agent("neuromorphic materials", k=2)
    print("Search hits:", ctx)
    recent = mem.retrieve_recent_summaries()
    print("Recent summaries:", recent)

if __name__ == "__main__":
    test_memory()
