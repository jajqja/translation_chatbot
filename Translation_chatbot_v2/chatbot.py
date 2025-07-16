from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from transformers import GPT2TokenizerFast
from typing import List, Dict, Any, Annotated, Union
from pydantic import BaseModel, Field
import nltk
from typing_extensions import TypedDict
import uuid

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Define data models
class KeywordPair(BaseModel):
    en: str = Field(description="Từ/cụm tiếng Anh")
    vi: str = Field(description="Bản dịch tiếng Việt đã dùng trong translation")

# Define the output model for translation results
class TranslationOutput(BaseModel):
    translation: str = Field(description="Văn bản đã dịch sang tiếng Việt")
    new_keywords: List[KeywordPair] = Field(
        description=(
            "Danh sách các từ khóa MỚI quan trọng (chưa có trong danh sách chuẩn). "
            "Nếu không có, trả về mảng rỗng."
        ),
        default_factory=list,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "translation": (
                    "Google đang mua lại Kaggle, một nền tảng tổ chức các cuộc thi "
                    "về khoa học dữ liệu và học máy …"
                ),
                "new_keywords": [
                    {"en": "Google", "vi": "Google"},
                    {"en": "Kaggle", "vi": "Kaggle"},
                    {"en": "machine learning", "vi": "học máy"},
                ],
            }
        }


# Define the state for the translation workflow
class TranslationState(TypedDict):
    """State for the translation workflow"""
    # Input
    original_text: str
    chunk_size: int
    domain: str
    style: str

    # Processing state
    chunks: List[str]
    current_chunk_index: int

    # Context management
    translated_context: str
    global_keywords: Dict[str, str]

    #new keywork
    new_keywords: Dict[str, str]

    # Results
    translations: List[str]
    final_translation: str

    # Messages
    messages: Annotated[List[Any], add_messages]


# Define the agent that will handle translation requests
class Agent:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: TranslationState):
        output: TranslationOutput = self.runnable.invoke(state)
        tool_call_id = str(uuid.uuid4())
        ai_msg = AIMessage(
            content="", 
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": "TranslationOutput",
                    "args": output.model_dump(),
                    "type": "function"
                }
            ]
        )

        return {"messages": [ai_msg]}

# Define the main chatbot class
class TranslationChatbot:
    """Chatbot for translating text in chunks using LangGraph and LLMs."""
    def __init__(self, llm_model="gpt-4o", temperature=1):
        self.prompt_template = build_translation_prompt
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.llm_runnable = self.prompt_template | self.llm.with_structured_output(TranslationOutput, method="function_calling")
        self.thread_id = str(uuid.uuid4())
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def initialize_chunks(state: TranslationState) -> Dict[str, Any]:
            """Initialize text chunks for processing"""
            chunks = split_chunk_text(state["original_text"], state["chunk_size"])
            return {
                "chunks": chunks,
                "current_chunk_index": 0,
                "translations": [],
                "translated_context": "",
                "messages": [],
                "final_translation": "",
                "chunk_results": ""
            }
        
        def should_continue(state: TranslationState) -> str:
            """Decision node to determine if we should continue processing"""
            if state["current_chunk_index"] >= len(state["chunks"]):
                return "combine"
            return "translator_chatbot"
        
        def process_output(state: TranslationState):
            last_msg = state["messages"][-1]

            # Check if the last message is an AIMessage with tool calls
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                tool_call = last_msg.tool_calls[0]
                args = tool_call.get("args", {})  # dict

                # Ensure args match TranslationOutput structure
                translation_output = TranslationOutput(**args)
            else:
                raise ValueError("Missing tool call in last AIMessage")

            # Update state fields
            result = translation_output.translation
            new_translations = state["translations"] + [result]
            kw = translation_output.new_keywords

            updated_keywords = (
                merge_keywords(state["global_keywords"], kw, strategy="keep_existing")
                if kw else state["global_keywords"]
            )

            translated_context = get_example_final_sentences(
                result,
                min_token_size=200
            ) if result else ""

            tool_msg = ToolMessage(
                tool_call_id=tool_call["id"],  
                content=translation_output
            )

            return {
                "chunks": state["chunks"],
                "final_translation": state["final_translation"],
                "current_chunk_index": state["current_chunk_index"] + 1,
                "translations": new_translations,
                "global_keywords": updated_keywords,
                "translated_context": translated_context,
                "messages": [tool_msg],
                "new_keywords": _list_to_dict(kw)
            }


        def finalize_translation(state: TranslationState) -> Dict[str, Any]:
            """Combine all translations into final result"""
            final_translation = "".join(state["translations"])
            
            return {
                "final_translation": final_translation,
                "messages": [AIMessage(content="Translation completed successfully.")],
                "global_keywords": state["global_keywords"],
            }
        
        # Build the graph
        workflow = StateGraph(TranslationState)
        
        # Add nodes
        workflow.add_node("initialize", initialize_chunks)
        workflow.add_node("translator_chatbot", Agent(self.llm_runnable))

        workflow.add_node("process_output", process_output)
        workflow.add_node("combine", finalize_translation)
        
        # Add edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "translator_chatbot")
        workflow.add_edge("translator_chatbot", "process_output")

        workflow.add_conditional_edges(
            "process_output",
            should_continue,
            {
                "translator_chatbot": "translator_chatbot",
                "combine": "combine"
            }
        )
        workflow.add_edge("combine", END)

        return workflow.compile(checkpointer=MemorySaver())
    
    # Stream translation method for streamlit app
    def stream_translate(
        self,
        text: str,
        chunk_size: int = 2000,
        domain: str = "general",
        style: str = "formal",
        kw: Dict[str, str] = None
    ):
        """ Stream translation progress using LangGraph for streamlit app."""
        initial_state = TranslationState(
            original_text=text,
            chunk_size=chunk_size,
            domain=domain,
            style=style,
            global_keywords=kw or {}
        )
        
        thread_config = {"configurable": {"thread_id": self.thread_id}}

        # Stream workflow execution
        for event in self.workflow.stream(initial_state, config=thread_config):
            node_name = list(event.keys())[0]
            state = event[node_name]

            if node_name == "process_output":
                translation = state["translations"]
                progress = state["current_chunk_index"]
                kw = state["new_keywords"]
                chunks_list = state["chunks"]

                yield {
                    "translated_chunk": translation[progress - 1],
                    "progress": progress,
                    "keywords": kw,
                    "global_keywords": state["global_keywords"],
                    "total_chunks": len(chunks_list),
                    "original_chunk": chunks_list[progress - 1]
                }
            if node_name == "combine":
                final_translation = state["final_translation"]
                global_keywords = state["global_keywords"]
                
                yield {
                    "final_translation": final_translation,
                    "global_keywords": global_keywords
                }

    def stream_event_translate(
        self,
        text: str,
        chunk_size: int = 2000,
        domain: str = "general",
        style: str = "formal",
        kw: Dict[str, str] = None
    ):
        """ Stream translation progress using LangGraph for streamlit app."""
        initial_state = TranslationState(
            original_text=text,
            chunk_size=chunk_size,
            domain=domain,
            style=style,
            global_keywords=kw or {}
        )
        
        thread_config = {"configurable": {"thread_id": self.thread_id}}

        # Stream workflow execution
        for event in self.workflow.stream(initial_state, config=thread_config):
            yield event

# Split text into chunks for processing
def split_chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=[".", "\n", "!", "?", ";", ":"],
        keep_separator=True,
    )
    return splitter.split_text(text)

# Get final sentences from chunk for context
def get_example_final_sentences(chunk: str, min_token_size: int = 300) -> str:
    """Get final sentences from chunk for context"""
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        sentences = nltk.sent_tokenize(chunk)
        selected_sentences = []
        total_tokens = 0

        for sentence in reversed(sentences):
            token_count = len(tokenizer.encode(sentence))
            selected_sentences.insert(0, sentence)
            total_tokens += token_count
            if total_tokens >= min_token_size:
                break

        return " ".join(selected_sentences)
    except Exception as e:
        print(f"Error in get_example_final_sentences: {e}")
        return chunk[:min_token_size]  # Fallback to character-based truncation

# Build the translation prompt
def build_translation_prompt(state: TranslationState) -> ChatPromptTemplate:
    """Build translation prompt"""
    # Create keyword string
    kw_str = "\n".join([f"- {k}: {v}" for k, v in state.get("global_keywords", {}).items()])
    if not kw_str:
        kw_str = "Chưa có thuật ngữ nào được xác định."
    
    # Get current chunk
    chunks = state.get("chunks", [])
    current_chunk_index = state.get("current_chunk_index", 0)
    
    if current_chunk_index >= len(chunks):
        current_chunk = ""
    else:
        current_chunk = chunks[current_chunk_index]
    
    # Get translated context
    translated_context = state.get("translated_context", "")
    if not translated_context:
        translated_context = "Đây là phần đầu tiên của văn bản."

    translation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""Bạn là một chuyên gia dịch thuật tiếng Anh sang tiếng Việt chuyên nghiệp với 10+ năm kinh nghiệm trong lĩnh vực {state.get('domain', 'general')}.

YÊU CẦU ĐỊNH DẠNG:
- Giữ nguyên vẹn cấu trúc văn bản gốc, không tự ý xóa hay thêm dấu câu bao gồm cả dấu chấm đầu dòng, ký hiệu đặc biệt, v.v. (ví dụ: `. I'm Minh Hoang` → `. Tôi là Minh Hoàng`)
- QUAN TRỌNG: Giữ nguyên CHÍNH XÁC tất cả dấu xuống dòng (\\n) trong văn bản gốc
- Giữ nguyên cấu trúc đoạn văn, danh sách, bullet points
- Không thêm hoặc bớt dấu xuống dòng so với bản gốc
- Phong cách dịch: {state.get('style', 'formal')}
    """
        ),
        (
            "user",
            f"""NGỮ CẢNH TRƯỚC ĐÓ:
Đây là phần cuối của bản dịch trước đó. Hãy đảm bảo tính liên kết và nhất quán: "{translated_context}"

THUẬT NGỮ CHUẨN:
Sử dụng CHÍNH XÁC các thuật ngữ sau (KHÔNG được thay đổi):
{kw_str}

NHIỆM VỤ:
1. Dịch đoạn văn dưới đây sang tiếng Việt
2. Tìm và liệt kê các từ khóa/thuật ngữ MỚI quan trọng cho ngữ cảnh (chưa có trong danh sách chuẩn)

ĐOẠN VĂN:
{current_chunk}

LƯU Ý: Nếu đoạn văn có từ khóa quan trọng, hãy LUÔN LUÔN liệt kê chúng trong phần keywords."""
        )
    ])

    return translation_prompt

# Utility function to convert list of KeywordPair to dict
def _list_to_dict(pairs: List[Union[KeywordPair, Dict[str, str]]]) -> Dict[str, str]:
    """Chuyển list KeywordPair → dict {en: vi}"""
    out: Dict[str, str] = {}
    for item in pairs:
        if isinstance(item, KeywordPair):
            out[item.en] = item.vi
        else:  # assume dict-like {"en":…, "vi":…}
            out[item["en"]] = item["vi"]
    return out

# Merge keywords with conflict resolution strategies
def merge_keywords(
    existing_keywords: Dict[str, str],
    new_keywords: Union[Dict[str, str], List[Union[KeywordPair, Dict[str, str]]]],
    strategy: str = "keep_existing"
) -> dict:
    """Merge keyword dictionaries"""
    merged = existing_keywords.copy()

    if isinstance(new_keywords, list):
        new_keywords = _list_to_dict(new_keywords)

    for eng, vi in new_keywords.items():
        if eng in merged:
            if merged[eng] == vi:
                continue
            if strategy == "overwrite":
                merged[eng] = vi
            elif strategy == "mark_conflict":
                merged[eng] = f"{merged[eng]} | CONFLICT with: {vi}"
        else:
            merged[eng] = vi

    return merged

# Usage example
if __name__ == "__main__":
    # Initialize NLTK data (run once)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Initialize chatbot
    chatbot = TranslationChatbot()
    
    # Example usage
    text = """
Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud Next conference in San Francisco this week, the official announcement could come as early as tomorrow. 

Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. 
Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, was founded by Goldbloom  and Ben Hamner in 2010? The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed to stay well ahead of them by focusing on its specific niche. 

The service is basically the de facto home for running data science and machine learning competitions. With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google will keep the service running - likely under its current name. While the acquisition is probably more about Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can share this code on the platform (the company previously called them 'scripts'). Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, Google chief economist Hal Varian, Khosla Ventures and Yuri Milner """
    
    try:
        # Translate with progress tracking
        print("Starting translation...")
        result = chatbot.stream_event_translate(
            text=text,
            chunk_size=800,
            domain="technical",
            style="formal",
            kw={
                "Google": "Google",
                "data scientist": "nhà khoa học dữ liệu",
                "Google Cloud Platform": "Nền tảng đám mây Google",
                "CEO": "Giám đốc điều hành",
            }
        )

        for event in result:
            print(f"Event: {event}")
            print("----------------------------------------------------------------------------")

        
        # print("Translation completed!")
        # print(f"Final translation: {result['final_translation']}")
        # print(f"Keywords discovered: {result['global_keywords']}")
        
            
    except Exception as e:
        print(f"Error during translation: {e}")
        import traceback
        traceback.print_exc()