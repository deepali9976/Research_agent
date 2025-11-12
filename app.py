import streamlit as st
from backend.agents.orchestrator import Orchestrator
import json
import os
from pathlib import Path
import time
import re
import base64

st.set_page_config(page_title="AI Research Agent", layout="wide", page_icon="🧠")

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
    /* Fix for HTML iframe content */
    iframe {
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Autonomous Research System by Deepali U</div>', unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size: 1.1rem; color: #666;'>
Multi-agent AI system that discovers domains, finds data, designs experiments, and writes research papers
</p>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    iterations = st.number_input("Research Iterations", min_value=1, max_value=10, value=1)
    show_logs = st.checkbox("Show Execution Logs", value=False)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button("🚀 Start Research", use_container_width=True, type="primary")

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'completed' not in st.session_state:
    st.session_state.completed = False

# Main execution
if start_button:
    st.session_state.running = True
    st.session_state.completed = False

if st.session_state.running:
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Log container
    if show_logs:
        log_expander = st.expander("📜 Execution Logs", expanded=False)
        log_placeholder = log_expander.empty()
    
    try:
        # Initialize orchestrator
        status_text.text("🚀 Initializing agents...")
        progress_bar.progress(10)
        orchestrator = Orchestrator()
        
        # Capture logs
        import io, sys
        log_stream = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = log_stream
        
        # Update progress as pipeline runs
        status_text.text("🌐 Discovering research domain...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        status_text.text("💭 Generating research questions...")
        progress_bar.progress(35)
        
        status_text.text("📊 Finding relevant datasets...")
        progress_bar.progress(50)
        
        status_text.text("⚗️ Designing experiments...")
        progress_bar.progress(65)
        
        status_text.text("🧠 Running critical review...")
        progress_bar.progress(80)
        
        status_text.text("📄 Writing research paper...")
        progress_bar.progress(90)
        
        # Run the actual research cycle
        result = orchestrator.run_cycle(iterations=iterations)
        
        # Restore stdout
        sys.stdout = old_stdout
        logs = log_stream.getvalue()
        
        # Update UI
        progress_bar.progress(100)
        status_text.text("✅ Research completed!")
        
        # Show logs if enabled
        if show_logs:
            log_placeholder.text_area("Logs", logs, height=300)
        
        st.session_state.running = False
        st.session_state.completed = True
        st.session_state.result = result
        st.session_state.logs = logs
        
    except Exception as e:
        sys.stdout = old_stdout
        st.error("❌ An error occurred during research")
        st.code(f"Error: {str(e)}")
        st.session_state.running = False

# Helper function to convert image to base64
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Helper function to fix HTML content
def fix_html_styling(html_content):
    """Add proper styling to HTML content for light background"""
    
    # Check if HTML already has styling
    if '<style>' not in html_content and '<head>' in html_content:
        # Insert style in head
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
        # Add full HTML structure
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

# Display results if completed
if st.session_state.completed:
    st.success("🎉 Research pipeline completed successfully!")
    
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
            
            # Extract image references
            image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            image_refs = re.findall(image_pattern, paper_content)
            
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
                            Path(img_path),  # Exact path
                            exp_dir / Path(img_path).name,  # Just filename in experiments
                            Path("backend/results/final_paper") / img_path,  # Relative to paper
                        ]
                        
                        for full_path in possible_paths:
                            if full_path.exists():
                                st.image(str(full_path), caption=caption or full_path.name, use_container_width=True)
                                break
                        else:
                            st.warning(f"⚠️ Image not found: {img_path}")
                else:
                    # Regular markdown content
                    if part.strip():
                        st.markdown(part, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Paper not found. Check if generation completed successfully.")
    
    with tab2:
        st.header("🖼️ All Visualizations")
        
        # Display all generated visualizations
        exp_dir = Path("backend/results/experiments")
        if exp_dir.exists():
            image_files = sorted(list(exp_dir.glob("*.png")) + list(exp_dir.glob("*.jpg")))
            
            if image_files:
                st.write(f"Found {len(image_files)} visualizations")
                
                # Add filter option
                show_all = st.checkbox("Show all images", value=False)
                
                if not show_all:
                    # Only show images referenced in the paper
                    paper_md = Path("backend/results/final_paper/mini_research_paper.md")
                    if paper_md.exists():
                        with open(paper_md, 'r', encoding='utf-8') as f:
                            paper_content = f.read()
                        
                        # Extract referenced image filenames
                        image_pattern = r'!\[.*?\]\(([^)]+)\)'
                        referenced = re.findall(image_pattern, paper_content)
                        referenced_names = {Path(p).name for p in referenced}
                        
                        # Filter images
                        image_files = [img for img in image_files if img.name in referenced_names]
                        st.info(f"Showing {len(image_files)} images referenced in the paper")
                
                cols = st.columns(3)
                for idx, img_path in enumerate(image_files):
                    with cols[idx % 3]:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)
                        
                        # Add download button for each image
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
            # Download paper (Markdown)
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
            # Download paper (HTML)
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
        
        # Download all visualizations as zip
        st.markdown("---")
        if st.button("📦 Package All Results as ZIP", use_container_width=True):
            import zipfile
            import io
            
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add paper files
                for paper_file in [paper_md, paper_html]:
                    if paper_file and paper_file.exists():
                        zip_file.write(paper_file, f"paper/{paper_file.name}")
                
                # Add all images
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