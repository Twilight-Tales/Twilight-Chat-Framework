import chainlit as cl
from chainlit.types import ThreadDict

from chatbot import setup_openai, setup_mistral, setup_llama, remove_matching_suffix, stop_tokens


from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable.config import RunnableConfig
import bookclub_backend as db
DB = db.DatabaseDriver()

from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

str_gpt4 = "GPT-4"
# str_mistral = "Mistral"
str_llama_7b = "Llama2_7B"
str_llama_7bq = "7B_GPTQ"
str_llama_13bq = "13B_GPTQ"

llm_dict = {
    str_gpt4: setup_openai,
    # str_mistral: setup_mistral,
    str_llama_7b: setup_llama,
    str_llama_7bq: setup_llama,
    str_llama_13bq: setup_llama
}

llm_args_dict = {
    str_gpt4: {'api_url': 'https://api.openai.com/v1/chat/completions', 'model_name': "gpt-4-1106-preview"},
    # str_mistral: {'api_url': 'MISTRAL_URL', 'model_name': 'MISTRAL_ID'},  # Uncomment and define if used
    str_llama_7b: {'api_url': "vLLM_URL", 'model_name': "Llama2_7B_ID"},
    str_llama_7bq: {'api_url': "vLLM_URL", 'model_name': "Llama2_7B_GPTQ_ID"},
    str_llama_13bq: {'api_url': "vLLM_URL", 'model_name': "Llama2_13B_GPTQ_ID"}
}


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name=str_gpt4,
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/100",
        ),
        cl.ChatProfile(
            name=str_llama_7b,
            markdown_description="The underlying LLM model is **Llama-7B-chat**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name=str_llama_7bq,
            markdown_description="The underlying LLM model is **Llama-7B-chat-GPTQ**.",
            icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name=str_llama_13bq,
            markdown_description="The underlying LLM model is **Llama-13B-chat-GPTQ**.",
            icon="https://picsum.photos/300",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    llm_choice = cl.user_session.get("chat_profile")
    if llm_choice == str_gpt4:
        memory = ConversationBufferMemory(memory_key="history", input_key="input",
                                          return_messages=True)
    else:
        memory = ConversationBufferMemory(memory_key="history", input_key="input",
                                          human_prefix="Elderly", ai_prefix="Host",
                                          return_messages=False)
    cl.user_session.set("memory", memory)
    llm_dict[llm_choice](**llm_args_dict[llm_choice])
@cl.on_chat_start
async def on_chat_start():
    llm = ChatOpenAI(streaming=True, temperature=0, model_name="gpt-4-1106-preview")
    template = """
Objective: You are the host of a bookclub that helps elderly people with dementia. 
You have to prompt them and see whether or not they fully understand the content of the book. 
Context will be provided and generate prompts based on that. 
Ask open-ended questions but make sure that a 5th grader could answer them. 
After a conversation has concluded, ask a question relating to the book again and then move on to the next chapter. 
Make sure that it prompts the user to want to read the next chapter. 
Tailor it to elderly people. 
Only ask one question per time and keep them less than or equal to 2 sentences. 
Pause and wait for the user to give a response to the question, then analyze the response given by the elderly person and provide feedback as well as the next question.

Book chapter:
{chapter_context}

Current conversation:
{chat_history}
Elderly: {input}
host:"""
    prompt = PromptTemplate(
        input_variables=["chat_history", "input", "chapter_context"], template=template
    )

    with open("books/alice.txt", "r", encoding='utf-8') as fp:
        book = fp.read()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"input": message.content.strip()},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    memory.chat_memory.add_user_message(message.content.strip())
    memory.chat_memory.add_ai_message(remove_matching_suffix(res.content.strip(), stop_tokens))


# @cl.password_auth_callback
# async def auth_callback(username: str, password: str):
#     # Fetch the user matching username from your database
#     # and compare the hashed password with the value stored in the database
#     success, user = DB.login_user(username, password)
#     if success:
#         return cl.User(
#             identifier=user.user_id, metadata={"role": "Patient", "provider": user.password}
#         )
#     else:
#         return None