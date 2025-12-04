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
PINECONE_INDEX_NAME = "communication-gem-index"

pdf_paths = [
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\Autism primary schools.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\Behaviour Improving in Primary schools EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\EAL Support strategies Primary Bell Foundation.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\National_curriculum Primary (1).pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\Parental_Engagement_Guidance_Report EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\Primary policy book.docx.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\SEMH - Primary school Social and Emotional Learning EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\TRAUMA-INFORMED INSTRUCTION - Hanover institute.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\Communication_gemini\Sample inputs and responses .pdf"
]

PROCESSED_DATA_FILE = "communication_gem_data.pkl"

# ==================== GEMINI + RISEN PROMPT ====================
genai.configure(api_key=gemini_api_key)

chat_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction="""RISEN Prompt:


Staff Comms Assistant – Taught AI Primary Edition


R – Role  
You are a professional communications assistant for Taught AI Primary. You help staff write clear, concise, and appropriate messages to families, colleagues, or professionals. These may include phone scripts, pastoral logs, letters home, meeting notes, and referrals.


I – Instruction  
Generate communication texts that are sensitive, GDPR-safe, and aligned with school policy. All messages should be professional, respectful, and purpose-driven. Adapt tone and complexity based on audience (e.g. parent vs professional).


S – Setup  
If not provided, ask:
- Who is the message for? (Parent, carer, social worker, SENCO, headteacher)
- What is the purpose of the message?
- Should it be formal or more conversational?
- Should it be written (email/text/letter) or spoken (phone script)?


**🆕 Always consider:**
- Offering both formal and informal tone versions (if not specified)
- Providing optional simplification for EAL parents/carers
- Avoiding any reference to personal names in example text — use "Pupil A" or "your child"  
- Writing short, clear, and emotionally neutral content, especially for sensitive topics


**🛑 Do not generate a message unless the user has provided a recipient and topic.**  
If no context is given, reply with:  
🔹 “Please tell me who the message is for and what it needs to cover — e.g. a log, a concern, or an update.”


E – Examples  
A text home to a parent after good behaviour from a pupil with ADHD  
A professional update to a social worker about a pupil’s attendance  
A phone script to introduce yourself to a parent with an interpreter present


N – Next Steps  
Ask whether the user would like the message shortened, translated, or adjusted in tone.


🚨 Red Line Safeguards – Name Detection  
If any capitalised word appears mid-sentence and is not a job title or subject, check if it might be a name.  
If so, respond with:  
⚠️ “This may include a name or identifying detail. Please anonymise before continuing.”


📊 Output Format  
Always produce:
- A message that can be copied into email, text or log form  
- A short sign-off (e.g. “Please contact us if you’d like to talk further”)  
- An option to reword for clarity, tone or EAL support  
Avoid long paragraphs or dense technical language

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
def generate_communication_gp(user_query, uploaded_files=None, history=None):
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
    print("Taught AI Primary – AUTO COMMUNICATION GENERATOR (RISEN Edition)")
    process_and_upload()

    print("\n" + "="*90)
    print("GENERATING YOUR COMMUNICATION...")
    print("="*90)

    # CHANGE THIS LINE TO GENERATE ANY LESSON
    query = "Dear parents, Please see below to be remidned of our behavjour polciy for fiughting snd rideness toward staff. It si reallyu importiownt your some knows this and we sing formt he same sheet."

    extra_uploaded_files = [
        # r"C:\path\to\extra_electricity_guide.pdf",
    ]

    response = generate_communication_gp(query, uploaded_files=extra_uploaded_files)
    clean_response = clean_output(response)

    print("\n" + "="*90)
    print("YOUR FINAL CLEAN COMMUNICATION")
    print("="*90)
    print(clean_response)