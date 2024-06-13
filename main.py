from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from create_database import OPENAI_KEY, CHROMA_PATH

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
    uvicorn.run(app, host="127.0.0.1", port=8000)