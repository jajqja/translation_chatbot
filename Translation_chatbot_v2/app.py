import streamlit as st
from typing import Dict, Optional
import PyPDF2
import io

# Import your chatbot
from chatbot import TranslationChatbot

# Page config
st.set_page_config(
    page_title="AI Translation Chatbot",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Custom CSS for better styling
st.markdown("""
<style>
    /* Header với gradient dịu hơn và text đậm */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #8EC5FC 0%, #E0C3FC 100%);
        color: #2c2c2c;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    
    .translation-container {
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        background-color: #fefefe;
        color: #1a1a1a; /* màu chữ đậm, dễ đọc */
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Từ khóa nổi bật */
    .keyword-badge {
        background-color: #f0f7ff;
        color: #1565c0;
        padding: 0.4rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
        display: inline-block;
        border: 1px solid #bbdefb;
    }
    
    /* Thông tin tiến độ */
    .progress-info {
        background-color: #f1f8f4;
        border-left: 5px solid #43a047;
        padding: 1rem;
        margin: 1.5rem 0;
        border-radius: 8px;
        color: #2e7d32;
        font-weight: 500;
    }
    
    /* Mỗi đoạn chunk đã dịch */
    .chunk-container {
        background-color: #fcfcfc;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.75rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)


def read_file_content(uploaded_file) -> Optional[str]:
    """Read content from uploaded file"""
    try:
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type in ["txt", "md"]:
            return str(uploaded_file.read(), "utf-8")

        elif file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        else:
            st.error("Unsupported file type!")
            return None
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def display_keywords(keywords: Dict[str, str], title: str):
    """Display keywords in a nice format"""
    if keywords:
        st.markdown(f"**{title}:**")
        keyword_html = ""
        for en, vi in keywords.items():
            keyword_html += f'<span class="keyword-badge">{en} → {vi}</span>'
        st.markdown(keyword_html, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 AI Translation Chatbot</h1>
        <p>Dịch thuật thông minh từ tiếng Anh sang tiếng Việt</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Model settings
        model_option = st.selectbox(
            "Mô hình AI:",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Độ sáng tạo của mô hình (0 = chính xác, 2 = sáng tạo)"
        )
        
        chunk_size = st.slider(
            "Kích thước chunk:",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000,
            help="Kích thước mỗi đoạn văn để dịch"
        )
        
        domain = st.selectbox(
            "Lĩnh vực:",
            ["general", "technical", "medical", "legal", "business", "academic"],
            index=0
        )
        
        style = st.selectbox(
            "Phong cách dịch:",
            ["formal", "informal", "academic", "casual"],
            index=0
        )
        
        # Initialize chatbot button
        if st.button("🔄 Khởi tạo Chatbot", type="primary"):
            with st.spinner("Đang khởi tạo chatbot..."):
                st.session_state.chatbot = TranslationChatbot(
                    llm_model=model_option,
                    temperature=temperature
                )
            st.success("Chatbot đã được khởi tạo!")
        
        # Predefined keywords
        st.subheader("📝 Từ khóa định sẵn")
        keywords_input = st.text_area(
            "Nhập từ khóa (format: en:vi, mỗi dòng một cặp):",
            placeholder="machine learning:học máy\nartificial intelligence:trí tuệ nhân tạo",
            height=100
        )

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Văn bản gốc")
        
        # Input method selection
        input_method = st.radio(
            "Chọn phương thức nhập:",
            ["Upload file", "Nhập text trực tiếp"],
            horizontal=True
        )
        
        text_to_translate = ""
        
        if input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Chọn file để dịch:",
                type=['txt', 'md', 'pdf'],
                help="Hỗ trợ file .txt, .md, .pdf"
            )
            
            if uploaded_file is not None:
                text_to_translate = read_file_content(uploaded_file)
                if text_to_translate:
                    st.success(f"✅ Đã đọc file: {uploaded_file.name}")
                    with st.expander("Xem nội dung file"):
                        st.text(text_to_translate[:1000] + "..." if len(text_to_translate) > 1000 else text_to_translate)
        
        else:
            text_to_translate = st.text_area(
                "Nhập văn bản tiếng Anh:",
                placeholder="Nhập văn bản cần dịch...",
                height=300
            )
        
        # Process keywords
        global_keywords = {}
        if keywords_input.strip():
            try:
                for line in keywords_input.strip().split('\n'):
                    if ':' in line:
                        en, vi = line.split(':', 1)
                        global_keywords[en.strip()] = vi.strip()
            except Exception as e:
                st.error(f"Lỗi parse từ khóa: {e}")
    
    with col2:
        st.subheader("🎯 Kết quả dịch")
        
        if st.session_state.chatbot is None:
            st.warning("⚠️ Vui lòng khởi tạo chatbot trước!")
        elif not text_to_translate:
            st.info("📝 Vui lòng nhập văn bản hoặc upload file để bắt đầu dịch")
        else:
            if st.button("🚀 Bắt đầu dịch", type="primary"):
                # Initialize session state for results
                if 'translation_results' not in st.session_state:
                    st.session_state.translation_results = {
                        'all_chunks': [],
                        'final_translation': "",
                        'final_keywords': {},
                        'is_completed': False
                    }
                
                # Reset results for new translation
                st.session_state.translation_results = {
                    'all_chunks': [],
                    'final_translation': "",
                    'final_keywords': {},
                    'is_completed': False
                }
                
                # Create containers for streaming results
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    keywords_display = st.empty()
                
                try:
                    # Start translation
                    for result in st.session_state.chatbot.stream_translate(
                        text=text_to_translate,
                        chunk_size=chunk_size,
                        domain=domain,
                        style=style,
                        kw=global_keywords
                    ):
                        if 'translated_chunk' in result:
                            # Update progress
                            progress = result['progress']
                            total_chunks = result['total_chunks']
                            progress_percentage = progress / total_chunks
                            
                            progress_bar.progress(progress_percentage)
                            progress_text.markdown(f"""
                            <div class="progress-info">
                                <strong>Đang dịch chunk {progress}/{total_chunks}</strong>
                                <br>Tiến độ: {progress_percentage:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add chunk to results
                            chunk_info = {
                                'original': result['original_chunk'],
                                'translated': result['translated_chunk'],
                                'keywords': result.get('keywords', {}),
                                'chunk_number': progress
                            }
                            st.session_state.translation_results['all_chunks'].append(chunk_info)
                            
                            # Display current keywords
                            if result.get('keywords'):
                                with keywords_display:
                                    display_keywords(result['keywords'], f"Từ khóa mới (Chunk {progress})")
                        
                        elif 'final_translation' in result:
                            st.session_state.translation_results['final_translation'] = result['final_translation']
                            st.session_state.translation_results['final_keywords'] = result.get('global_keywords', {})
                            st.session_state.translation_results['is_completed'] = True
                    
                    # Display final results
                    progress_bar.progress(1.0)
                    progress_text.markdown("""
                    <div class="progress-info">
                        <strong>✅ Hoàn thành dịch!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.translation_history.append({
                        'original': text_to_translate[:100] + "..." if len(text_to_translate) > 100 else text_to_translate,
                        'translated': st.session_state.translation_results['final_translation'],
                        'keywords': st.session_state.translation_results['final_keywords'],
                        'settings': {
                            'model': model_option,
                            'domain': domain,
                            'style': style,
                            'chunk_size': chunk_size
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Lỗi trong quá trình dịch: {str(e)}")
                    st.exception(e)

    # Display results in tabs (only if translation is completed)
    if 'translation_results' in st.session_state and st.session_state.translation_results['is_completed']:
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📝 Bản dịch hoàn chỉnh", "🔍 Theo từng Chunk", "🔑 Từ khóa & Tải xuống"])
        
        with tab1:
            st.subheader("📋 Bản dịch hoàn chỉnh")
            
            # Display original text
            with st.expander("👀 Xem văn bản gốc", expanded=False):
                st.text_area("Văn bản tiếng Anh:", value=text_to_translate, height=200, disabled=True)
            
            # Display translated text
            st.markdown("**Bản dịch:**")
            st.markdown(f"""
            <div class="translation-container">
                {st.session_state.translation_results['final_translation'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            # Copy to clipboard button (using text area)
            st.text_area(
                "Sao chép bản dịch:",
                value=st.session_state.translation_results['final_translation'],
                height=300,
                key="copy_translation"
            )
        
        with tab2:
            st.subheader("🔍 Dịch theo từng Chunk")
            
            if st.session_state.translation_results['all_chunks']:
                for i, chunk in enumerate(st.session_state.translation_results['all_chunks']):
                    with st.expander(f"Chunk {chunk['chunk_number']} ({len(chunk['original'])} ký tự)", expanded=False):
                        col_orig, col_trans = st.columns([1, 1])
                        
                        with col_orig:
                            st.markdown("**Văn bản gốc:**")
                            st.markdown(f"""
                            <div style="background-color: #f0f0f0; padding: 1rem; border-radius: 5px; max-height: 300px; overflow-y: auto; color: #1a1a1a;">
                                {chunk['original'].replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_trans:
                            st.markdown("**Bản dịch:**")
                            st.markdown(f"""
                            <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 5px; max-height: 300px; overflow-y: auto; color: #1a1a1a;">
                                {chunk['translated'].replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show keywords for this chunk
                        if chunk['keywords']:
                            st.markdown("**Từ khóa trong chunk này:**")
                            display_keywords(chunk['keywords'], "")
            else:
                st.info("Chưa có chunk nào được dịch.")
        
        with tab3:
            st.subheader("🔑 Từ khóa & Tải xuống")
            
            # Display all keywords
            if st.session_state.translation_results['final_keywords']:
                display_keywords(st.session_state.translation_results['final_keywords'], "Tất cả từ khóa đã sử dụng")
                
                # Export keywords
                keywords_text = "\n".join([f"{en}: {vi}" for en, vi in st.session_state.translation_results['final_keywords'].items()])
                st.download_button(
                    label="📥 Tải xuống từ khóa",
                    data=keywords_text,
                    file_name="keywords.txt",
                    mime="text/plain"
                )
            else:
                st.info("Không có từ khóa nào được ghi nhận.")
            
            st.markdown("---")
            
            # Download options
            st.subheader("💾 Tải xuống")
            
            col_download1, col_download2 = st.columns([1, 1])
            
            with col_download1:
                st.download_button(
                    label="📄 Tải xuống bản dịch",
                    data=st.session_state.translation_results['final_translation'],
                    file_name="translation.txt",
                    mime="text/plain"
                )
            
            with col_download2:
                # Create combined report
                report_content = f"""BÁO CÁO DỊCH THUẬT
{'='*50}

VĂN BẢN GỐC:
{text_to_translate}

{'='*50}

BẢN DỊCH:
{st.session_state.translation_results['final_translation']}

{'='*50}

TỪ KHÓA SỬ DỤNG:
{chr(10).join([f"- {en}: {vi}" for en, vi in st.session_state.translation_results['final_keywords'].items()])}

{'='*50}

CÀI ĐẶT:
- Model: {model_option}
- Domain: {domain}
- Style: {style}
- Chunk size: {chunk_size}
"""
                
                st.download_button(
                    label="📋 Tải xuống báo cáo đầy đủ",
                    data=report_content,
                    file_name="translation_report.txt",
                    mime="text/plain"
                )

    # Translation history
    if st.session_state.translation_history:
        st.subheader("📚 Lịch sử dịch")
        with st.expander("Xem lịch sử dịch"):
            for i, item in enumerate(reversed(st.session_state.translation_history)):
                st.markdown(f"""
                **Lần dịch #{len(st.session_state.translation_history) - i}**
                - Gốc: {item['original']}
                - Cài đặt: {item['settings']['model']} | {item['settings']['domain']} | {item['settings']['style']}
                """)
                with st.expander(f"Xem chi tiết lần dịch #{len(st.session_state.translation_history) - i}"):
                    st.text_area("Bản dịch:", value=item['translated'], height=200, key=f"history_{i}")
                    if item['keywords']:
                        display_keywords(item['keywords'], "Từ khóa")

if __name__ == "__main__":
    main()