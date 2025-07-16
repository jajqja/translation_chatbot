from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2TokenizerFast
import nltk
import json
import re

# Initialize NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Function to split text into chunks
def split_chunk_text(text: str, chunk_size: int) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=[".", "\n", "!", "?",";",":"],
        keep_separator=True,
    )

    return splitter.split_text(text)

# Function to get example final sentences from a chunk
def get_example_final_sentences(chunk: str, min_token_size: int = 200) -> str:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
 
    sentences = nltk.sent_tokenize(chunk)
    selected_sentences = []
    total_tokens = 0

    for sentence in reversed(sentences):
        token_count = len(tokenizer.encode(sentence))
        selected_sentences.insert(0, sentence)  # chèn vào đầu để giữ đúng thứ tự
        total_tokens += token_count
        if total_tokens >= min_token_size:
            break

    example = " ".join(selected_sentences)

    return example

# Function to parse LLM JSON output
def parse_llm_json_output(output: str) -> tuple[str, dict]:
    try:
        # Remove any code block formatting
        output = re.sub(r'```json\s*|\s*```', '', output.strip())
        
        data = json.loads(output)
        return data.get("translation", ""), data.get("keywords", {})
    except json.JSONDecodeError:
        # Fallback
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("translation", ""), data.get("keywords", {})
            except json.JSONDecodeError:
                pass
        
        print("Lỗi: LLM không trả về JSON hợp lệ.")
        return "", {}


def build_translation_prompt(
    chunk: str,
    translated_context: str = "",
    keyword_dict: dict = None,
    domain: str = "general",
    style: str = "formal"
) -> str:
    """
    Create an input prompt for LLM to translate a text chunk with context and keyword support.

    Args:
        chunk (str): The text chunk to be translated.
        translated_context (str): A few final sentences of the previous chunk's translation (if any).
        keyword_dict (dict): Dictionary of translated keywords for LLM to maintain consistency.
        domain (str): Domain (general, technical, medical, legal, etc.)
        style (str): Style (formal, casual, academic)

    Returns:
        str: Complete prompt.
    """
    prompt_parts = []

    # 1. Giới thiệu vai trò
    prompt_parts.append(f"""Bạn là một chuyên gia dịch thuật tiếng Anh sang tiếng Việt chuyên nghiệp với 10+ năm kinh nghiệm trong lĩnh vực {domain}.

YÊU CẦU ĐỊNH DẠNG:
- Giữ nguyên vẹn cấu trúc văn bản gốc, bao gồm cả dấu chấm đầu dòng, ký hiệu đặc biệt, v.v. (ví dụ: `. I'm Minh Hoang` → `. Tôi là Minh Hoàng`)
- QUAN TRỌNG: Giữ nguyên CHÍNH XÁC tất cả dấu xuống dòng (\\n) trong văn bản gốc
- Nếu văn bản gốc có 1 dòng trống (\\n\\n), bản dịch cũng phải có 1 dòng trống
- Nếu văn bản gốc có 2 dòng trống (\\n\\n\\n), bản dịch cũng phải có 2 dòng trống
- Giữ nguyên cấu trúc đoạn văn, danh sách, bullet points
- Không thêm hoặc bớt dấu xuống dòng so với bản gốc""")
    
    # 2. Nếu có ngữ cảnh trước
    if translated_context:
        prompt_parts.append(f"""NGỮ CẢNH TRƯỚC ĐÓ:
Đây là phần cuối của bản dịch trước đó. Hãy đảm bảo tính liên kết và nhất quán:
"{translated_context.strip()}" 
""")

    # 3. Nếu có từ khóa
    if keyword_dict:
        kw_str = """THUẬT NGỮ CHUẨN:
Sử dụng CHÍNH XÁC các thuật ngữ sau (KHÔNG được thay đổi):"""
        for en, vi in keyword_dict.items():
            kw_str += f"\n- {en}: {vi}"

        prompt_parts.append(kw_str)

    # 4. Phần chính: yêu cầu dịch chunk hiện tại
    prompt_parts.append(f"""NHIỆM VỤ:
1. Dịch đoạn văn dưới đây sang tiếng Việt
3. Nếu phát hiện các từ khóa/thuật ngữ mới CHƯA CÓ trong danh sách:
   - CHỈ thêm nếu thực sự quan trọng cho ngữ cảnh, chuyên ngành, hoặc dễ gây hiểu nhầm nếu dịch không nhất quán.
   - Không cần thêm những từ phổ thông, từ đã được dịch đúng mà không ảnh hưởng đến ngữ nghĩa.
   - Bạn hoàn toàn có thể trả về danh sách từ khóa rỗng nếu thật sự cảm thấy không có từ khóa mới nào cần thêm.
3. Trả về kết quả theo định dạng JSON sau:

{{
"translation": "<bản dịch hoàn chỉnh>",
"keywords": {{
    "<từ tiếng Anh>": "<cụm từ tiếng Việt đã dùng trong phần 'translation' ở trên, KHÔNG được dịch lại>",
    ...
}}
}}

LƯU Ý QUAN TRỌNG:
- Chỉ liệt kê các thuật ngữ MỚI (chưa có trong danh sách chuẩn)
- Ưu tiên thuật ngữ chuyên ngành, khái niệm quan trọng
- Không liệt kê từ thông thường như "the", "and", "very"...

ĐOẠN VĂN CẦN DỊCH:
{chunk.strip()}""")

    return "\n\n".join(prompt_parts)


def merge_keywords(
    existing_keywords: dict,
    new_keywords: dict,
    strategy: str = "keep_existing"  # options: "keep_existing", "overwrite", "mark_conflict"
) -> dict:
    """
    Merge two keyword dictionaries: existing keywords and new keywords from LLM.
    
    Args:
        existing_keywords (dict): Existing keywords (e.g., {"neural network": "mạng nơ-ron"}).
        new_keywords (dict): Newly detected keywords from the current chunk.
        strategy (str): Strategy for handling translation conflicts:
            - "keep_existing": keep the old translation
            - "overwrite": overwrite with the new translation
            - "mark_conflict": keep the old translation and add a note if different

    Returns:
        dict: The merged keyword dictionary.
    """
    merged = existing_keywords.copy()

    for eng, vi in new_keywords.items():
        if eng in merged:
            if merged[eng] == vi:
                continue  
            if strategy == "overwrite":
                merged[eng] = vi
            elif strategy == "mark_conflict":
                merged[eng] = f"{merged[eng]} | CONFLICT with: {vi}"
            # default: "keep_existing" → giữ nguyên
        else:
            merged[eng] = vi

    return merged
