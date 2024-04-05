from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Your existing imports for the chain
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.chains import create_retrieval_chain

app = Flask(__name__)

# Load environment variables
load_dotenv()

# def get_documents_from_web(url):
#     loader = WebBaseLoader(url) #will scrape the webpage and add the content of the page to a langchain document.
#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 20) #still not doing it. Use a vector database to do a semantic search or similarity search. USed to store the documents. Use a function to pass the query. 
#     splitDocs = splitter.split_documents(docs)
#     return splitDocs

# def create_db(docs):
#     embedding = OpenAIEmbeddings()
#     vectorStore = Chroma.from_documents(docs, embedding= embedding)
#     return vectorStore

def create_chain():
    llm = ChatOpenAI(
    model = "gpt-3.5-turbo-1106",
    temperature= 0.4,
    max_tokens= 1000,
    )

    prompt = ChatPromptTemplate.from_template(
        """
    You are a helpful assistant:
    context : {context}
    Question: {input}
    """
    )

    chain = create_stuff_documents_chain(
    llm = llm,
    prompt = prompt

    )

    # retriever = vectorStore.as_retriever( search_kwargs = {"k": 3})
    # retrieval_chain = create_retrieval_chain(
    #     retriever= retriever, # will fetch most relevant documents fromt the vector store and pass it to a {context} variable in the prompt
    #     combine_docs_chain= chain,

    # )


    # return retrieval_chain
    return chain

# Initialize your chain here (simplified example, adjust according to your needs)
# def initialize_chain():
#     # Example URL - replace with actual URL or mechanism to fetch documents
#     url = "https://python.langchain.com/docs/expression_language/"
#     docs = get_documents_from_web(url)
#     vectorStore = create_db(docs)
#     chain = create_chain(vectorStore)
#     return chain

# Initialize the chain once when the app starts
# chain = initialize_chain()
chain = create_chain()

def process_chat(chain, question):
    # This function was in your original script but not included in your Flask app script.
    # It uses the chain to process the question and returns the response.
    response = chain.invoke({
        "input": question,
    })

    return response["answer"]



@app.route('/process_chat', methods=['POST'])
def process_chat_endpoint():
    # Extract question from the request
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Process the question through your chain
    try:
        response = process_chat(chain, question)
        return jsonify({'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
