import streamlit as st
from langsmith import Client
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langsmith.evaluation import LangChainStringEvaluator
import openai
from langsmith import evaluate

# Ititialize streamlit variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load environment variables from a .env file
load_dotenv()

# Create a new LangSmith client
client = Client()

# Define dataset: these are your test cases
dataset_name = "QA Example Dataset"
dataset = client.create_dataset(dataset_name)
client.create_examples(
    inputs=[
        {"question": "What is LangChain?"},
        {"question": "What is LangSmith?"},
        {"question": "What is OpenAI?"},
        {"question": "What is Google?"},
        {"question": "What is Mistral?"},
    ],
    outputs=[
        {"answer": "A framework for building LLM applications"},
        {"answer": "A platform for observing and evaluating LLM applications"},
        {"answer": "A company that creates Large Language Models"},
        {"answer": "A technology company known for search"},
        {"answer": "A company that creates Large Language Models"},
    ],
    dataset_id=dataset.id,
)

# Define a prompt template for grading answers
_PROMPT_TEMPLATE = """You are an expert professor specialized in grading students' answers to questions.
You are grading the following question:
{query}
Here is the real answer:
{answer}
You are grading the following predicted answer:
{result}
Respond with CORRECT or INCORRECT:
Grade:
"""

# Create a PromptTemplate object with the specified input variables and template
PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE
)

# Initialize a ChatOpenAI model for evaluation
eval_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create a QA evaluator
qa_evaluator = LangChainStringEvaluator("qa", config={"llm": eval_llm, "prompt": PROMPT})

# Initialize an OpenAI client
openai_client = openai.Client()

# Generate a response to the question using OpenAI
def my_app(question):
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Respond to the users question in a short, concise manner (one short sentence)."
            },
            {
                "role": "user",
                "content": question,
            }
        ],
    ).choices[0].message.content

def langsmith_app(inputs):
    # Get the output from my_app for the given input question
    output = my_app(inputs["question"])
    #st.session_state.chat_history.append(output)
    return {"output": output}

# Evaluate the AI system using the specified data and evaluators
experiment_results = evaluate(
    langsmith_app, # Your AI system
    data=dataset_name, # The data to predict and grade over
    evaluators=[qa_evaluator], # The evaluators to score the results
    experiment_prefix="openai-3.5", # A prefix for your experiment names to easily identify them
)
