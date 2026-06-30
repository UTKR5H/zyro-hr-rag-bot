import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖")

# Set this under Streamlit Cloud -> App settings -> Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

# Same refusal text + threshold used in the Kaggle notebook —
# keep these in sync if you tune the threshold there.
REFUSAL_MESSAGE = "I can only answer HR-related questions from Zyro Dynamics policy documents."
RELEVANCE_THRESHOLD = 1.0  # MUST match the value that worked in Cell 12 of the notebook

SYSTEM_PROMPT = (
    "You are the Zyro Dynamics HR Help Desk assistant. "
    "Answer ONLY using the provided context from internal HR policy documents. "
    "Be concise and professional, and ground every claim in the context. "
    "Do not mention 'context' or 'documents' explicitly — answer naturally."
)


@st.cache_resource(show_spinner="Loading HR knowledge base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # The "hr_faiss_index" folder (downloaded from Kaggle Cell 8 output)
    # must sit right next to this app.py file in the repo.
    return FAISS.load_local("hr_faiss_index", embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def get_chain(_vectorstore):
    retriever = _vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512, api_key=GROQ_API_KEY)

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    chain = prompt | llm | StrOutputParser()
    return retriever, chain, format_docs


def ask_bot(question, vectorstore, retriever, chain, format_docs):
    # Wrapped in try/except — the app must NEVER crash to a blank screen.
    try:
        scored_docs = vectorstore.similarity_search_with_score(question, k=4)
        if not scored_docs or scored_docs[0][1] > RELEVANCE_THRESHOLD:
            return REFUSAL_MESSAGE
        docs = retriever.invoke(question)
        context = format_docs(docs)
        return chain.invoke({"context": context, "question": question})
    except Exception as e:
        return f"Sorry, I hit a technical issue answering that. Please try again. ({type(e).__name__})"


# ---------------- UI ----------------
st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Ask me about leave policy, payroll, benefits, and compliance.")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it under App settings → Secrets on Streamlit Cloud.")
    st.stop()

vectorstore = load_vectorstore()
retriever, chain, format_docs = get_chain(vectorstore)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Checking HR policy docs..."):
            answer = ask_bot(prompt, vectorstore, retriever, chain, format_docs)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})