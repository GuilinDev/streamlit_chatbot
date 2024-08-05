import streamlit as st

from pydantic import BaseModel

# LLM chain
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage

model = ChatOllama(model_name='llama3.1', streaming=True)
mermory_key = 'history'

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name=mermory_key),
        ('human', '{input}')
    ]
)


class Message(BaseModel):
    content: str
    role: str


if "message" not in st.session_state:
    st.session_state.messages = []


def to_message_placeholder(messages):
    return [
        AIMessage(content=message['content']) if message['role'] == "ai" else HumanMessage(content=message.content)
        for message in messages
    ]


chain = {
            'input': lambda x: x['input'],
            'history': lambda x: to_message_placeholder(x['messages'])
        } | prompt | model | StrOutputParser()

# left side for chat content, right side for history
left, right = st.columns(0.7, 0.3)

with left:
    # chat content
    container = st.container
    with container:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.write(message['content'])

    # receive users input, save into session_state
    prompt = st.chat_input("Hello, what I can do for you?")
    if prompt:
        st.session_state.messages.append(Message(content=prompt, role="human").model_dump())
        with container:
            with st.chat_message("human"):
                st.write(prompt)

        # get LLM response and display
        with container:
            response = st.write_stream(chain.stream({'input': prompt, 'messages': st.session_state.messages}))

        st.session_state.messages.append(Message(content=response, role='ai').model_dump())

with right:
    # display chat history
    st.json(st.session_state.messages)
