from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from create_database import OPENAI_KEY, CHROMA_PATH, generate_data_store
import uvicorn
import os

app = FastAPI()

# Ensure the OpenAI API key is set
if not OPENAI_KEY:
    raise ValueError("The OpenAI API key is not set in the environment variables.")

# Load the existing Chroma database
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
try:
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
except Exception as e:
    raise ValueError(f"Unable to load the database: {e}")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class Question(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/ask")
def ask_question(question: Question):
    query_text = question.question
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if not results or results[0][1] < 0.7:
        raise HTTPException(status_code=404, detail="Unable to find matching results.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(openai_api_key=OPENAI_KEY)
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = {
        "response": response_text,
        "sources": sources
    }
    return formatted_response

if __name__ == "__main__":
    generate_data_store()
    uvicorn.run(app, host="127.0.0.1", port=5500)
