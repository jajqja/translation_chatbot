import streamlit as st
import json
from typing import Dict, List
import pandas as pd
import fitz 

# Import your modules
from chatbot import Chatbot
from chunkprocess import (
    split_chunk_text,
    build_translation_prompt,
    parse_llm_json_output,
    merge_keywords,
    get_example_final_sentences
)

# Page configuration
st.set_page_config(
    page_title="AI Translation Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chunk-container {
        background-color: #f9f9fb; /* Nền trắng */
        color: #1e293b;            /* Chữ đậm dễ đọc */
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .translation-result {
        background-color: #e6f4ea; /* Nền xanh nhạt */
        color: #1a3c34;            /* Chữ xanh đậm */
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .keywords-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'global_keywords' not in st.session_state:
        st.session_state.global_keywords = {}
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    if 'current_translation' not in st.session_state:
        st.session_state.current_translation = None

def display_keywords(keywords: Dict, title: str = "Keywords"):
    """Display keywords in a formatted way"""
    if keywords:
        st.markdown(f"**{title}:**")
        cols = st.columns(3)
        for i, (en, vi) in enumerate(keywords.items()):
            with cols[i % 3]:
                st.markdown(f"• `{en}`: {vi}")

# Khởi tạo chatbot 1 lần
@st.cache_resource
def load_chatbot():
    return Chatbot()

chatbot = load_chatbot()

def main():
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 AI Translation Assistant</h1>
        <p>Intelligent English to Vietnamese Translation with Context & Terminology Management</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Translation settings
        st.subheader("Translation Settings")
        
        domain = st.selectbox(
            "Domain",
            ["general", "technical", "medical", "legal", "business", "academic"],
            index=0
        )
        
        style = st.selectbox(
            "Style",
            ["formal", "casual", "academic"],
            index=0
        )
        
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=800,
            max_value=1000,
            value=800,
            step=50
        )
        
        
        # Global keywords management
        st.subheader("Danh sách Keywords")
        if st.session_state.global_keywords:
            st.write(f"Tổng số keywords: {len(st.session_state.global_keywords)}")
            if st.button("Clear Keywords"):
                st.session_state.global_keywords = {}
                st.success("Keywords cleared!")
        
        # Export keywords
        if st.session_state.global_keywords:
            keywords_json = json.dumps(st.session_state.global_keywords, indent=2, ensure_ascii=False)
            st.download_button(
                label="Tải về Keywords",
                data=keywords_json,
                file_name="translation_keywords.json",
                mime="application/json"
            )
    
    # Main content area
    st.header("Nhập văn bản hoặc tải lên file")

    # Text input methods
    input_method = st.radio(
        "Chọn phương thức nhập liệu",
        ["Nhập trực tiếp", "Tải file từ máy"],
        horizontal=True
    )
    
    input_text = ""

    if input_method == "Nhập trực tiếp":
        input_text = st.text_area(
            "Nhập văn bản cần dịch:",
            height=300,
            placeholder="Paste your English text here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Tải file văn bản (.txt, .md, .pdf):",
            type=['txt', 'md', 'pdf']
        )

        input_text = ""

        if uploaded_file:
            file_type = uploaded_file.name.split(".")[-1].lower()

            if file_type in ["txt", "md"]:
                input_text = str(uploaded_file.read(), "utf-8")

            elif file_type == "pdf":
                pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text_chunks = [page.get_text() for page in pdf_doc]
                input_text = "\n\n".join(text_chunks)
    
    # Manual keywords input
    st.subheader("Nhập Keywords thủ công (Optional)")
    manual_keywords = st.text_area(
        "Nhập keywords (định dạng JSON):",
        placeholder='{"machine learning": "học máy", "algorithm": "thuật toán"}',
        height=100
    )
    
    if manual_keywords:
        try:
            manual_kw_dict = json.loads(manual_keywords)
            st.session_state.global_keywords.update(manual_kw_dict)
            st.success(f"Added {len(manual_kw_dict)} keywords")
        except json.JSONDecodeError:
            st.error("Không thể phân tích cú pháp JSON. Vui lòng kiểm tra định dạng.")

    st.subheader("Dịch văn bản")

    if st.button("Dịch văn bản", type="primary", disabled=not input_text):
        if not input_text:
            st.warning("Vui lòng nhập văn bản cần dịch")
        else:
            translate_text(input_text, domain, style, chunk_size)
    
    # Display current translation
    if st.session_state.current_translation:
        st.divider()
        display_translation_results(st.session_state.current_translation)
    
    # Translation history
    if st.session_state.translation_history:
        st.header("📚 Lịch sử")
        
        with st.expander("Xem lịch sử dịch thuật", expanded=True):
            for i, translation in enumerate(reversed(st.session_state.translation_history)):
                st.markdown(f"**Dịch thuật {len(st.session_state.translation_history) - i}**")
                st.write(f"**Chunks:** {translation['chunks']}")
                st.write(f"**Tổng số Keywords:** {len(translation['global_keywords'])}")
                st.markdown("---")

def translate_text(text: str, domain: str, style: str, chunk_size: int):
    """Main translation function"""
    try:
        # Split text into chunks
        chunks = split_chunk_text(text, chunk_size)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {
            "chunks": len(chunks),
            "translations": [],
            "global_keywords": st.session_state.global_keywords.copy()
        }
        
        translated_context = ""
        
        for i, chunk in enumerate(chunks):
            status_text.text(f"Dịch chunk {i+1}/{len(chunks)}...")
            
            # Build prompt
            prompt = build_translation_prompt(
                chunk, 
                translated_context, 
                results["global_keywords"],
                domain,
                style
            )
            
            # Get translation from LLM
            llm_response = chatbot.generate(prompt)
            translation, keywords = parse_llm_json_output(llm_response)
            
            # Merge keywords
            results["global_keywords"] = merge_keywords(
                results["global_keywords"], 
                keywords
            )
            
            # Update context for next chunk
            translated_context = get_example_final_sentences(translation)
            
            results["translations"].append({
                "chunk_index": i,
                "original": chunk,
                "translation": translation,
                "keywords": keywords,
                "prompt": prompt
            })
            
            progress_bar.progress((i + 1) / len(chunks))
        
        # Update global keywords
        st.session_state.global_keywords = results["global_keywords"]
        
        # Store results
        st.session_state.current_translation = results
        st.session_state.translation_history.append(results)
        
        status_text.text("Đã dịch thành công!")
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"Lỗi dịch: {str(e)}")

def display_translation_results(results: Dict):
    """Display translation results"""
    st.subheader("Kết quả Dịch Thuật")
    
    # Summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks", results["chunks"])
    with col2:
        st.metric("Total Keywords", len(results["global_keywords"]))
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Bản dịch", "Theo từng đoạn", "Từ khóa"])
    
    with tab1:
        st.markdown("### Bản dịch hoàn chỉnh")
        full_translation = "".join([t["translation"] for t in results["translations"]])
        st.markdown(f'<div class="translation-result">{full_translation}</div>', unsafe_allow_html=True)
        
        # Download option
        st.download_button(
            label="Tải xuống Bản dịch",
            data=full_translation,
            file_name="translation.txt",
            mime="text/plain"
        )
    
    with tab2:
        st.markdown("### Theo từng đoạn")
        for i, chunk_result in enumerate(results["translations"]):
            with st.expander(f"Chunk {i+1}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Bản gốc:**")
                    st.markdown(f'<div class="chunk-container">{chunk_result["original"]}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Bản dịch:**")
                    st.markdown(f'<div class="translation-result">{chunk_result["translation"]}</div>', unsafe_allow_html=True)
                
                if chunk_result["keywords"]:
                    display_keywords(chunk_result["keywords"], "New Keywords in This Chunk")

    with tab3:
        st.markdown("### Từ điển Từ khóa")
        if results["global_keywords"]:
            # Search keywords
            search_term = st.text_input("Tìm kiếm từ khóa:")
            
            filtered_keywords = results["global_keywords"]
            if search_term:
                filtered_keywords = {
                    k: v for k, v in results["global_keywords"].items() 
                    if search_term.lower() in k.lower() or search_term.lower() in v.lower()
                }
            
            # Display as table
            if filtered_keywords:
                df = pd.DataFrame(list(filtered_keywords.items()), columns=["English", "Vietnamese"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Không tìm thấy từ khóa nào phù hợp với tìm kiếm")
        else:
            st.info("Chưa có từ khóa nào được trích xuất")

if __name__ == "__main__":
    main()