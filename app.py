import streamlit as st
from backend.agents.orchestrator import Orchestrator
import json
import os
from pathlib import Path
import time
import re
import base64
import io
import sys
import traceback
from threading import Thread
import queue

st.set_page_config(page_title="AI Research Agent", layout="wide", page_icon="🧠")

# ===== INITIALIZE SESSION STATE FIRST =====
if 'running' not in st.session_state:
    st.session_state.running = False
if 'completed' not in st.session_state:
    st.session_state.completed = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'logs' not in st.session_state:
    st.session_state.logs = ""
if 'error' not in st.session_state:
    st.session_state.error = None
    
def cleanup_results():
    """Clean up results from previous runs"""
    results_dir = Path("backend/results")
    cleaned_count = 0
    
    for folder in ["experiments", "final_paper"]:
        folder_path = results_dir / folder
        if folder_path.exists():
            for file in folder_path.glob("*"):
                if file.is_file():
                    try:
                        file.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Could not delete {file}: {e}")
        else:
            folder_path.mkdir(parents=True, exist_ok=True)
    
    return cleaned_count

def get_image_base64(image_path):
    """Convert image to base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def fix_html_styling(html_content):
    """Add proper styling to HTML content"""
    if '<style>' not in html_content and '<head>' in html_content:
        style_tag = """
        <style>
            body {
                background-color: white !important;
                color: #262730 !important;
                font-family: 'Source Sans Pro', sans-serif;
                padding: 20px;
                line-height: 1.6;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #262730 !important;
            }
            img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px auto;
            }
            pre, code {
                background-color: #f0f2f6;
                color: #262730;
            }
        </style>
        """
        html_content = html_content.replace('</head>', f'{style_tag}</head>')
    elif '<style>' not in html_content:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    background-color: black !important;
                    color: white !important;
                    font-family: 'Source Sans Pro', sans-serif;
                    padding: 20px;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: white !important;
                    margin-top: 24px;
                    margin-bottom: 16px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }}
                pre, code {{
                    background-color: #f0f2f6;
                    color: #262730;
                    padding: 10px;
                    border-radius: 4px;
                }}
                p {{
                    margin-bottom: 16px;
                }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
    return html_content

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    iframe {
        background-color: white !important;
    }
    .log-box {
        background-color: #0e1117;
        color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Autonomous Research System</div>', unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size: 1.1rem; color: #666;'>
Multi-agent AI system that discovers domains, finds data, designs experiments, and writes research papers
</p>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    iterations = st.number_input("Research Iterations", min_value=1, max_value=10, value=1)
    show_logs = st.checkbox("Show Real-time Logs", value=True)
    timeout_minutes = st.number_input("Timeout (minutes)", min_value=1, max_value=60, value=15)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Previous Results", use_container_width=True):
        count = cleanup_results()
        st.success(f"✅ Cleaned {count} files!")
        time.sleep(1)
        st.rerun()
    
    st.markdown("---")
    st.info("💡 **Tip**: Enable real-time logs to monitor agent activity")

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button("🚀 Start Research", use_container_width=True, type="primary")
    
    if st.session_state.running:
        if st.button("⏹️ Stop Research", use_container_width=True, type="secondary"):
            st.session_state.running = False
            st.warning("⚠️ Research stopped by user")
            st.rerun()

# Main execution
if start_button:
    st.session_state.running = True
    st.session_state.completed = False
    st.session_state.error = None

if st.session_state.running:
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Auto-cleanup before starting
    status_text.text("🧹 Preparing workspace...")
    cleanup_results()
    
    # Real-time log display
    if show_logs:
        st.markdown("### 📜 Live Execution Logs")
        log_container = st.container()
        log_display = log_container.empty()
    
    # Initialize log capture BEFORE try block
    log_stream = io.StringIO()
    old_stdout = sys.stdout
    
    # Track start time for timeout
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    try:
        # Initialize orchestrator
        status_text.text("🚀 Initializing agents...")
        progress_bar.progress(5)
        orchestrator = Orchestrator()
        
        # Redirect stdout to capture logs
        sys.stdout = log_stream
        
        # Create a wrapper to run with timeout monitoring
        def run_research():
            return orchestrator.run_cycle(iterations=iterations)
        
        # Progress stages
        stages = [
            (10, "🌐 Discovering research domain..."),
            (25, "💭 Generating research questions..."),
            (40, "📊 Finding relevant datasets..."),
            (55, "⚗️ Designing experiments..."),
            (70, "🧪 Running experiments..."),
            (85, "🧠 Running critical review..."),
            (95, "📄 Writing research paper...")
        ]
        
        stage_idx = 0
        result = None
        
        # Start the research in a separate thread to allow monitoring
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def research_worker():
            try:
                res = run_research()
                result_queue.put(res)
            except Exception as e:
                error_queue.put(e)
        
        worker_thread = Thread(target=research_worker, daemon=True)
        worker_thread.start()
        
        # Monitor progress with timeout
        while worker_thread.is_alive():
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                sys.stdout = old_stdout
                st.error(f"⏱️ Research timed out after {timeout_minutes} minutes")
                st.warning("The agents may be stuck in a loop. Try reducing iterations or check your agent code.")
                st.session_state.running = False
                st.stop()
            
            # Update progress
            if stage_idx < len(stages):
                progress, message = stages[stage_idx]
                progress_bar.progress(progress)
                status_text.text(f"{message} ({int(elapsed/60)}m {int(elapsed%60)}s elapsed)")
                stage_idx = (stage_idx + 1) % len(stages)
            
            # Show logs
            if show_logs:
                current_logs = log_stream.getvalue()
                if current_logs:
                    # Show last 30 lines
                    lines = current_logs.split('\n')[-30:]
                    log_display.markdown(f'<div class="log-box">{"<br>".join(lines)}</div>', unsafe_allow_html=True)
            
            time.sleep(2)
        
        # Get result or error
        if not error_queue.empty():
            raise error_queue.get()
        
        if not result_queue.empty():
            result = result_queue.get()
        
        # Restore stdout
        sys.stdout = old_stdout
        logs = log_stream.getvalue()
        
        # Update UI
        progress_bar.progress(100)
        status_text.text("✅ Research completed!")
        
        st.session_state.running = False
        st.session_state.completed = True
        st.session_state.result = result
        st.session_state.logs = logs
        
        # Auto-refresh to show results
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        # Restore stdout safely
        sys.stdout = old_stdout
        st.error("❌ An error occurred during research")
        st.code(f"Error: {str(e)}")
        
        with st.expander("📋 Full Traceback"):
            st.code(traceback.format_exc())
        
        with st.expander("📜 Execution Logs"):
            st.code(log_stream.getvalue())
        
        st.session_state.running = False
        st.session_state.error = str(e)

# Display results if completed
if st.session_state.completed and not st.session_state.running:
    st.success("🎉 Research pipeline completed successfully!")
    
    # Show execution summary
    if st.session_state.logs:
        with st.expander("📊 Execution Summary"):
            log_lines = st.session_state.logs.split('\n')
            st.metric("Total Log Lines", len(log_lines))
            st.text_area("Full Logs", st.session_state.logs, height=200)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📄 Research Paper", "🖼️ Visualizations", "💾 Downloads"])
    
    with tab1:
        st.header("📄 Generated Research Paper")
        
        # Find the paper
        paper_md = Path("backend/results/final_paper/mini_research_paper.md")
        paper_html = Path("backend/results/final_paper/mini_research_paper.html")
        
        # Try HTML first
        if paper_html.exists():
            with open(paper_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Fix styling
            html_content = fix_html_styling(html_content)
            
            # Convert image paths to base64 if they're local references
            exp_dir = Path("backend/results/experiments")
            if exp_dir.exists():
                # Find all image references in HTML
                img_pattern = r'<img[^>]+src="([^"]+)"'
                
                def replace_img_src(match):
                    src = match.group(1)
                    # If it's a relative path, convert to base64
                    if not src.startswith(('http://', 'https://', 'data:')):
                        img_path = exp_dir / Path(src).name
                        if img_path.exists():
                            img_base64 = get_image_base64(img_path)
                            if img_base64:
                                ext = img_path.suffix.lower()
                                mime = 'image/png' if ext == '.png' else 'image/jpeg'
                                return match.group(0).replace(src, f'data:{mime};base64,{img_base64}')
                    return match.group(0)
                
                html_content = re.sub(img_pattern, replace_img_src, html_content)
            
            # Display using iframe
            import streamlit.components.v1 as components
            components.html(html_content, height=1200, scrolling=True)
        
        # Fallback to markdown
        elif paper_md.exists():
            with open(paper_md, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            
            # Add custom CSS for better markdown rendering
            st.markdown("""
            <style>
                div[data-testid="stMarkdownContainer"] h1 { 
                    margin-top: 2rem !important; 
                    margin-bottom: 1rem !important; 
                }
                div[data-testid="stMarkdownContainer"] h2 { 
                    margin-top: 1.5rem !important; 
                    margin-bottom: 0.75rem !important; 
                }
                div[data-testid="stMarkdownContainer"] h3 { 
                    margin-top: 1.25rem !important; 
                    margin-bottom: 0.5rem !important; 
                }
                div[data-testid="stMarkdownContainer"] h4 {
                    margin-top: 1rem !important;
                    margin-bottom: 0.5rem !important;
                }
                div[data-testid="stMarkdownContainer"] p {
                    margin-bottom: 1rem !important;
                    line-height: 1.6 !important;
                }
                div[data-testid="stMarkdownContainer"] ul, 
                div[data-testid="stMarkdownContainer"] ol {
                    margin-bottom: 1rem !important;
                    margin-top: 0.5rem !important;
                }
                div[data-testid="stMarkdownContainer"] li {
                    margin-bottom: 0.25rem !important;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Split content by images and display
            parts = re.split(r'(!\[.*?\]\(.*?\))', paper_content)
            
            for part in parts:
                if part.startswith('!['):
                    # Extract image info
                    match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', part)
                    if match:
                        caption, img_path = match.groups()
                        
                        # Find the actual image file
                        exp_dir = Path("backend/results/experiments")
                        possible_paths = [
                            Path(img_path),
                            exp_dir / Path(img_path).name,
                            Path("backend/results/final_paper") / img_path,
                        ]
                        
                        for full_path in possible_paths:
                            if full_path.exists():
                                st.image(str(full_path), caption=caption or full_path.name, use_container_width=True)
                                break
                        else:
                            st.warning(f"⚠️ Image not found: {img_path}")
                else:
                    if part.strip():
                        st.markdown(part)
        else:
            st.warning("⚠️ Paper not found. Check if generation completed successfully.")
            st.info("💡 The research may have completed but failed to generate the final paper. Check the logs above.")
    
    with tab2:
        st.header("🖼️ All Visualizations")
        
        exp_dir = Path("backend/results/experiments")
        if exp_dir.exists():
            image_files = sorted(list(exp_dir.glob("*.png")) + list(exp_dir.glob("*.jpg")))
            
            if image_files:
                st.write(f"Found {len(image_files)} visualizations")
                
                show_all = st.checkbox("Show all images", value=False)
                
                if not show_all:
                    paper_md = Path("backend/results/final_paper/mini_research_paper.md")
                    if paper_md.exists():
                        with open(paper_md, 'r', encoding='utf-8') as f:
                            paper_content = f.read()
                        
                        image_pattern = r'!\[.*?\]\(([^)]+)\)'
                        referenced = re.findall(image_pattern, paper_content)
                        referenced_names = {Path(p).name for p in referenced}
                        
                        image_files = [img for img in image_files if img.name in referenced_names]
                        st.info(f"Showing {len(image_files)} images referenced in the paper")
                
                cols = st.columns(3)
                for idx, img_path in enumerate(image_files):
                    with cols[idx % 3]:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)
                        
                        with open(img_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download",
                                data=file,
                                file_name=img_path.name,
                                mime=f"image/{img_path.suffix[1:]}",
                                key=f"download_{img_path.name}"
                            )
            else:
                st.info("No visualizations generated yet")
        else:
            st.info("No experiments folder found")
    
    with tab3:
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            paper_md = Path("backend/results/final_paper/mini_research_paper.md")
            if paper_md.exists():
                with open(paper_md, 'rb') as f:
                    st.download_button(
                        "📄 Download Paper (Markdown)",
                        f,
                        file_name="research_paper.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
        
        with col2:
            paper_html = Path("backend/results/final_paper/mini_research_paper.html")
            if paper_html.exists():
                with open(paper_html, 'rb') as f:
                    st.download_button(
                        "🌐 Download Paper (HTML)",
                        f,
                        file_name="research_paper.html",
                        mime="text/html",
                        use_container_width=True
                    )
        
        st.markdown("---")
        if st.button("📦 Package All Results as ZIP", use_container_width=True):
            import zipfile
            
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for paper_file in [paper_md, paper_html]:
                    if paper_file and paper_file.exists():
                        zip_file.write(paper_file, f"paper/{paper_file.name}")
                
                exp_dir = Path("backend/results/experiments")
                if exp_dir.exists():
                    for img_file in exp_dir.glob("*.png"):
                        zip_file.write(img_file, f"visualizations/{img_file.name}")
                    for img_file in exp_dir.glob("*.jpg"):
                        zip_file.write(img_file, f"visualizations/{img_file.name}")
            
            zip_buffer.seek(0)
            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_buffer,
                file_name="research_results.zip",
                mime="application/zip",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 Powered by Multi-Agent AI System</p>
    <p style='font-size: 0.8rem;'>Autonomous research through intelligent agent coordination</p>
</div>
""", unsafe_allow_html=True)