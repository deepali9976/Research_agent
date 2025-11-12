#To check groq integration
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env file if present

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
response = llm.invoke("Write a 2-line summary of what LangChain is.")
print(response)
