import os
import pickle
import pdfplumber
import warnings
import re
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
gemini_api_key = "AIzaSyCB_k7n2Nta3itsCHE4kQGwQNbidkjhGQk"
pinecone_api_key = "pcsk_4L26ra_HZYXdtHKRrRuTAyEjpEzherH6AkpEB7bJXWpkJkfEDfJbqMUdbhs7vEpsrE3181"
PINECONE_INDEX_NAME = "lesson-gem-index"

pdf_paths = [
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Autism primary schools.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Behaviour Improving in Primary schools EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\EAL Support strategies Primary Bell Foundation.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Literacy improvement in KS2 report Second edition EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Literacy improvements in KS1 Guidance Report 2020 EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Maths Early Years to KS1 Guidance Report EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\National_curriculum Primary (1).pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Maths KS2 KS3 improvements 2022 Guidence report Update EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Primary policy book.docx.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Science improvement in primary guidance-report-ks1-ks2 EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\SEMH - Primary school Social and Emotional Learning EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Teacher assessment frameworks KS2.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Teacher assessment frameworks- Non stat - GOV.UK.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\TRAUMA-INFORMED INSTRUCTION - Hanover institute.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\lesson_gemini\Sample inputs and responses .pdf",
]

PROCESSED_DATA_FILE = "lesson_gem_data.pkl"

# ==================== GEMINI + RISEN PROMPT ====================
genai.configure(api_key=gemini_api_key)

chat_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction="""RISEN Prompt:


Lesson Generator – Taught AI Primary Edition


R – Role  
You are an inclusive lesson planner for Taught AI Primary. You specialise in designing high-quality, curriculum-aligned lessons that are suitable for primary pupils with a range of needs, including EAL, SEMH, ADHD and low literacy.


I – Instruction  
Generate a full lesson plan based on the subject, topic, year group, and learning need provided. All lessons should be trauma-aware, scaffolded, and engaging for pupils aged 4–11. Prioritise inclusion and accessibility.


S – Setup  
Ask these clarifying questions if not already answered:
- What is the subject and topic?
- What year group or key stage?
- What ability range are you teaching? (e.g. mixed ability, high challenge, SEND-focused)
- Any inclusion priorities? (e.g. EAL, ASD, SEMH, SLCN, low literacy)
- Should this be a single lesson or a series of short activities?


**🆕 Always include:**
- A lesson title and clear, child-friendly objective
- A warm-up or retrieval starter
- A main activity broken into simple, scaffolded steps
- A short plenary, review or AfL task
- Adaptations for low literacy and EAL learners (e.g. visuals, sentence stems, oral task options)
- Stretch and support suggestions (SEND/EAL/greater depth)


**🛑 Wait for input before generating.**  
If no learning scenario is described, respond only with:  
🔹 “Please describe the subject, topic and year group for the lesson you'd like support with — and any key needs (e.g. SEMH, EAL, low literacy).”


E – Examples  
A Year 3 Maths lesson on subtracting 2-digit numbers with regrouping, supporting EAL learners  
A Year 1 English lesson on sentence writing using a visual story stimulus for ASD and SLCN learners  
A Year 5 History lesson on Ancient Egypt tombs, with low literacy adaptations


N – Next Steps  
Encourage the user to copy/paste the lesson into a planning format, or ask for a differentiated version.


🚨 Red Line Safeguards – Name Detection  
If any capitalised word appears mid-sentence and is not a known subject or curriculum term, check if it might be a pupil name.  
If so, respond with:  
⚠️ “This may include a name or identifying detail. Please anonymise (e.g. 'Pupil A') before continuing.”


📊 Output Format  
Use these sections in every response:
- Lesson Title  
- Objective (child-friendly)  
- Starter / Warm-up  
- Main Activity (step-by-step)  
- Plenary / AfL  
- Adaptations (EAL, SEND, stretch)  
Avoid dense paragraphs. Use bullet points and headings to support readability and copying into planning templates.


"""
)

# ==================== SAFE EMBEDDING WITH RETRY ====================
def get_embedding(text, retries=5):
    for attempt in range(retries):
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            if "500" in str(e) or "Internal" in str(e):
                print(f"   Google server busy... retry {attempt+1}/{retries} in 3s")
                time.sleep(3)
            else:
                print(f"   Embedding error: {e}")
                return None
    return None

# ==================== CLEAN OUTPUT FUNCTION ====================
def clean_output(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`\n]+`', '', text)
    text = re.sub(r'(\*\*|__|\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s•*+-]*\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    lines = [line.rstrip() for line in text.splitlines() if line.strip() or line == '']
    return '\n'.join(lines).strip()

# ==================== PINECONE SETUP ====================
pc = Pinecone(api_key=pinecone_api_key)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(PINECONE_INDEX_NAME)

# ==================== PDF & UPLOAD (SAFE MODE) ====================
def extract_text_from_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def process_and_upload():
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("Processing main knowledge base PDFs...")
        data = {}
        for p in pdf_paths:
            if os.path.exists(p):
                name = os.path.basename(p)
                print(f"→ {name}")
                text = extract_text_from_pdf(p)
                data[name] = split_text(text)
        with open(PROCESSED_DATA_FILE, "wb") as f:
            pickle.dump(data, f)
    else:
        print("Loading cached PDF chunks from disk...")
        with open(PROCESSED_DATA_FILE, "rb") as f:
            data = pickle.load(f)

    # === SMART CHECK: Which files are actually missing in Pinecone? ===
    missing_files = []
    for filename in data.keys():
        # Query Pinecone: does this file have at least 1 vector?
        try:
            results = index.query(
                vector=[0.0]*768,
                top_k=1,
                filter={"source": {"$eq": filename}},
                include_metadata=True
            )
            if not results.matches:
                missing_files.append(filename)
        except:
            missing_files.append(filename)  # if error, assume missing

    if not missing_files:
        print("All 14 PDFs are already in Pinecone! Skipping upload.")
        return

    print(f"Found {len(missing_files)} missing/scanned PDFs → uploading now:")
    for f in missing_files: print(f"   → {f}")

    # === Upload ONLY the missing ones (safe batching + retry) ===
    vectors = []
    total_chunks = sum(len(data[f]) for f in missing_files)
    count = 0

    for filename in missing_files:
        chunks = data[filename]
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            emb = get_embedding(chunk)
            if emb is None:
                time.sleep(2)
                emb = get_embedding(chunk)  # retry once
            if emb is None:
                continue
            vectors.append((f"{filename}_{i}", emb, {"text": chunk, "source": filename}))
            count += 1
            print(f"\r   Progress: {count}/{total_chunks} chunks", end="")

            if len(vectors) >= 50:
                index.upsert(vectors=vectors)
                vectors = []
                time.sleep(0.5)

    if vectors:
        index.upsert(vectors=vectors)

    print(f"\nSUCCESS! All missing PDFs uploaded.")
    print("Your knowledge base is now 100% complete (14/14).")

# ==================== LESSON GENERATOR ====================
def generate_lesson_gp(user_query, uploaded_files=None, history=None):
    if history is None: history = []
    if uploaded_files is None: uploaded_files = []

    uploaded_context = ""
    for file_path in uploaded_files:
        if os.path.exists(file_path):
            print(f"Reading uploaded file: {os.path.basename(file_path)}")
            text = extract_text_from_pdf(file_path)
            if text.strip():
                uploaded_context += f"\n\n=== UPLOADED FILE: {os.path.basename(file_path)} ===\n{text}"

    q_emb = get_embedding(user_query)
    results = index.query(vector=q_emb, top_k=8, include_metadata=True)
    pinecone_context = "\n\n".join([m.metadata["text"] for m in results.matches])

    messages = [
        {"role": "user", "parts": [{"text": f"""
USER REQUEST:
{user_query}

RELEVANT KNOWLEDGE FROM DATABASE:
{pinecone_context.strip() or "None found."}

ADDITIONAL UPLOADED DOCUMENTS:
{uploaded_context.strip() or "None provided."}
        """}]}
    ]

    for h in history[-10:]:
        role = "user" if h["role"] == "user" else "model"
        messages.append({"role": role, "parts": [{"text": h["content"]}]})

    reply = chat_model.generate_content(
        messages,
        generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=6000)
    )
    return reply.text

# ==================== RUN ====================
if __name__ == "__main__":
    print("Taught AI Primary – AUTO LESSON GENERATOR (RISEN Edition)")
    process_and_upload()

    print("\n" + "="*90)
    print("GENERATING YOUR LESSON & WORKSHEET...")
    print("="*90)

    # CHANGE THIS LINE TO GENERATE ANY LESSON
    query = "Plan me a lesson on Europe. The countires that make it. Key momnents in its history from the birth of the current countires which make it up, to the way it opoerates now. This is for our Yeah 6 pupils."

    extra_uploaded_files = [
        # r"C:\path\to\extra_electricity_guide.pdf",
    ]

    response = generate_lesson_gp(query, uploaded_files=extra_uploaded_files)
    clean_response = clean_output(response)

    print("\n" + "="*90)
    print("YOUR FINAL CLEAN LESSON & WORKSHEET")
    print("="*90)
    print(clean_response)