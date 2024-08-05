import os
import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-pro", streaming=True)

memory_key = 'history'
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name=memory_key),
    ('human', '{input}')
])

class Message(BaseModel):
    content: str
    role: str

if "messages" not in st.session_state:
    st.session_state.messages = []

def to_message_placeholder(messages):
    return [
        AIMessage(content=message['content']) if message['role'] == "ai" else HumanMessage(content=message['content'])
        for message in messages
    ]

chain = {
    'input': lambda x: x['input'],
    'history': lambda x: to_message_placeholder(x['messages'])
} | prompt | llm | StrOutputParser()

# Use st.columns() with a list of relative widths
left, right = st.columns([0.7, 0.3])

with left:
    # chat content
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # receive user's input, save into session_state
    user_input = st.chat_input("Hello, what can I do for you?")
    if user_input:
        st.session_state.messages.append(Message(content=user_input, role="human").model_dump())
        with st.chat_message("human"):
            st.write(user_input)

        # get LLM response and display
        with st.chat_message("ai"):
            response = st.write_stream(chain.stream({'input': user_input, 'messages': st.session_state.messages}))

        st.session_state.messages.append(Message(content=response, role='ai').model_dump())

with right:
    # display chat history
    st.json(st.session_state.messages)
