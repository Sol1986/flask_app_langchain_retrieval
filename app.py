from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Your existing imports for the chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Load environment variables
load_dotenv()

def create_chain():
    llm = ChatOpenAI(
    model = "gpt-3.5-turbo-1106",
    temperature= 0.4,
    max_tokens= 1000,
    )

    prompt = ChatPromptTemplate.from_template(
       """
        system: You are a helpful assistant.
        question: {input}

       """
    )

    chain = prompt | llm 

    return chain


chain = create_chain()

def process_chat(chain, question):
    # This function was in your original script but not included in your Flask app script.
    # It uses the chain to process the question and returns the response.
    response = chain.invoke({
        "input": question,
    })

    return response.content



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





# while True:
#         user_input = input("You: ")
#         if user_input.lower() =='exit':
#             break
#         response = process_chat(chain, user_input)
#         print("Assistant: ", response)