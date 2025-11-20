# %%
import os
import streamlit as st
from groq import Groq
from pypdf import PdfReader   
from dotenv import load_dotenv

# %%
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.1-8b-instant"   

# %%
# -------------------------------------------------
# 1. Load all pages from your 4 PDFs (runs only once)
# -------------------------------------------------
@st.cache_resource
def load_pdfs():
    chunks = []
    folder = "pdfs"   
    
    if not os.path.exists(folder):
        st.error("Create a folder called 'pdfs' and put your 4 PDFs there!")
        st.stop()
    
    for file in os.listdir(folder):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, file)
        try:
            reader = PdfReader(path)
            st.write(f"Loading {file} ({len(reader.pages)} pages)")
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "source": f"{file} – page {i}",
                        "text": text.strip()
                    })
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
    
    st.success(f"Loaded {len(chunks)} pages from your PDFs")
    return chunks

# Load once at startup
all_chunks = load_pdfs()

# -------------------------------------------------
# 2. Find the best pages for the question (simple & works great for 4 PDFs)
# -------------------------------------------------
def find_best_pages(question, top_k=5):
    if not all_chunks:
        return []
    
    q_words = set(question.lower().split())
    scored = []
    
    for chunk in all_chunks:
        chunk_words = set(chunk["text"].lower().split())
        matches = len(q_words & chunk_words)
        if matches > 0:
            scored.append((matches, chunk))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored[:top_k]]

# -------------------------------------------------
# 3. Talk to Groq using the real PDF pages as context
# -------------------------------------------------
def get_answer(question):
    pages = find_best_pages(question, top_k=5)
    
    if not pages:
        return "I couldn't find anything about that in your PDFs."
    
    context = "\n\n".join(f"From {p['source']}:\n{p['text']}" for p in pages)
    
    prompt = f"""You are an excellent computer science teacher.
Use the following pages from the student's PDFs to answer the question and offer simplified explanations of those answers from the note.
If the answer is not there, say "Not found in the PDFs".

PDF PAGES:
{context}

QUESTION: {question}

Answer step-by-step and end with a real-life analogy."""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1500
    )
    return response.choices[0].message.content, pages

# 4. Chat history + UI
# -------------------------------------------------
st.title("Algorithm & Complexity Tutor")
st.caption("Answers come only from your 4 uploaded PDFs")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Algorithms & Complexity tutor. Ask me anything from your PDFs."}
    ]

# Display all previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if user_input := st.chat_input("Ask me anything from your notes..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # Get answer from PDFs + Groq
    with st.chat_message("assistant"):
        with st.spinner("Searching your PDFs + answering..."):
            answer, sources = get_answer(user_input)
        st.write(answer)
        
        # Save assistant reply to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Show sources
        if sources:
            with st.expander(f"Sources ({len(sources)} pages)"):
                for s in sources:
                    st.caption(s["source"])
    
    # Important: rerun so the new messages appear immediately
    st.rerun()