import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from create_database import OPENAI_KEY, CHROMA_PATH

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser() # create parser for command line arguments
    parser.add_argument("query_text", type=str, help="The query text.") # add argument for query text
    args = parser.parse_args() # parse the arguments
    query_text = args.query_text # get the query text from the arguments

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_KEY) # create an instance of OpenAIEmbeddings
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    except Exception as e:
        print(f"Unable to load database: {e}")
        return

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3) # search the database for the query text and get the top 3 results
    if len(results) == 0 or results[0][1] < 0.7: # if there are no results or the top result has a score less than 0.7
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI(openai_api_key=OPENAI_KEY)
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
