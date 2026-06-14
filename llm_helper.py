from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Configure the LLM model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

if __name__ == "__main__":
    response = llm.invoke("How many hours in 100 years")
    print(response.content)