import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🏢"
)

st.title("🏢 Zyro Dynamics HR Assistant")

CORPUS_PATH = "zyro-dynamics-hr-corpus"

@st.cache_resource
def initialize_rag():

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512
    )

    return retriever, llm

retriever, llm = initialize_rag()

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are the Zyro Dynamics HR Assistant.

Answer ONLY using the provided HR policy context.

If the answer cannot be found in the context, say:

"I couldn't find that information in the Zyro Dynamics HR policies."

Context:
{context}

Question:
{question}

Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are an HR query classifier.

Determine if the following question is related to:

- HR policies
- leave
- payroll
- benefits
- onboarding
- performance reviews
- conduct
- compliance
- travel expenses
- IT policy
- work from home

Question:
{question}

Respond ONLY with:

YES

or

NO
""")

REFUSAL_MESSAGE = (
    "I can only answer questions related to Zyro Dynamics HR policies."
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    prompt = RAG_PROMPT.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(prompt)

    return response.content

def ask_bot(question):

    check_prompt = OOS_PROMPT.invoke({
        "question": question
    })

    decision = llm.invoke(
        check_prompt
    ).content.strip().upper()

    if "NO" in decision:
        return {
            "answer": REFUSAL_MESSAGE
        }

    answer = rag_chain(question)

    return {
        "answer": answer
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask a question about Zyro Dynamics HR policies..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    result = ask_bot(question)

    answer = result["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })