from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Chatbot:
    """ A simple chatbot class that uses OpenAI's Chat API to generate responses. """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 1):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate(self, prompt: str) -> str:
        """ Generate a response from the LLM based on the provided prompt."""
        response = self.llm.invoke(prompt)
        return response.content if response else ""

