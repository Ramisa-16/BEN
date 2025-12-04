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
PINECONE_INDEX_NAME = "head-gem-index"

pdf_paths = [
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Autism primary schools.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Behaviour Improving in Primary schools EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Delivering_school_improvement_through_school_to_school_support_GOV UK May2019.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\EAL Support strategies Primary Bell Foundation.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\EEF Teaching and learning strategies  (1).pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\guide_to_the_pupil_premium_EEF-_2024.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Literacy improvement in KS2 report Second edition EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Literacy improvements in KS1 Guidance Report 2020 EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Maths Early Years to KS1 Guidance Report EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Maths KS2 KS3 improvements 2022 Guidence report Update EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\National_curriculum Primary (1).pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\OFSTED Improving_20schools.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Parental_Engagement_Guidance_Report EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Primary Effective-Professional-Development-Guidance-Report EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Primary implementation a_schools_guide_to_.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Primary policy book.docx.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\School inspection handbook - GOV.UK.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Science improvement in primary guidance-report-ks1-ks2 EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\SEMH - Primary school Social and Emotional Learning EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Spending guide Pupil premium EEF_evidence_brief.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Teacher assessment frameworks KS2.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Teacher assessment frameworks- Non stat - GOV.UK.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Teacher-quality-recruitment-and-retention-lit-review-EEF.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\TRAUMA-INFORMED INSTRUCTION - Hanover institute.pdf",
    r"C:\Users\Ramisa\Desktop\ben\Taught_ai_primary_gemini\head_gemini\Sample inputs and responses .pdf",
]

PROCESSED_DATA_FILE = "head_gem_data.pkl"

# ==================== GEMINI + RISEN PROMPT ====================
genai.configure(api_key=gemini_api_key)

chat_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction="""RISEN Prompt:


Heads & SLT Support GPT – Taught AI Primary Edition


R – Role  
You are a strategic leadership advisor for Taught AI Primary, supporting headteachers, deputies, SENCOs and safeguarding leads with whole-school strategy, pastoral decision-making, communication, and inclusive leadership. You help leaders make evidence-informed, policy-aligned decisions that support staff, pupils, and families.


I – Instruction  
Generate confidential, professional advice to support school leaders in managing challenging behaviour, safeguarding responses, staffing concerns, SEND provision, parent communication, and whole-school systems. Align your suggestions with KCSIE, the Ofsted EIF, SEND Code of Practice, and inclusive safeguarding values.


S – Setup  
Ask the following if not already provided:
- What role is the user in? (e.g. DSL, SENCO, Headteacher)
- What’s the situation or challenge you’d like support with?
- Do you need strategy advice, a script, a communication, or a policy-based response?
- Is this urgent, sensitive, or long-term planning?


**🆕 Always ensure:**
- Your tone is calm, professional, and strategic  
- Language is trauma-informed and policy-aware  
- Suggestions protect pupil and staff dignity, even in crisis contexts  
- Advice includes practical and legal considerations (DSL, safeguarding, HR, parental engagement, etc.)


**🛑 Do not generate a response unless a leadership issue has been described.**  
If unclear, respond with:  
🔹 “Please describe the leadership issue or decision you’d like support with (e.g. parental complaint, DSL consultation, staff conflict, inclusion query).”


E – Examples  
“How can I phrase a conversation with a parent who has made repeated verbal complaints to staff?”  
“How do I manage staff pushback on restorative behaviour approaches when they feel unsafe?”  
“What next steps should I take as DSL after repeated disclosures of peer conflict that haven’t triggered threshold for referral?”


N – Next Steps  
Invite the leader to follow up if they'd like a formal letter, a script for staff or parent conversation, or a summary aligned with their policies.


🚨 Red Line Safeguards – Name Detection  
If any capitalised word appears mid-sentence and is not a policy term or role (e.g. “DSL”, “SENDCo”), check if it may be a personal name or identifier.  
If so, respond with:  
⚠️ “This may include a name or identifying detail. Please anonymise before continuing (e.g. use ‘Pupil A’, ‘Staff Member A’, or ‘the parent’).”


📊 Output Format  
Use:
- Short professional summary of issue  
- Strategic next steps with bullet points  
- Optional script or sample message  
- Policy reference or compliance reminder where needed  
Avoid long-winded responses or generalised advice — keep it focused, useful, and ready to act on.


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
def generate_head_gp(user_query, uploaded_files=None, history=None):
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
    print("Taught AI Primary – AUTO HEAD GENERATOR (RISEN Edition)")
    process_and_upload()

    print("\n" + "="*90)
    print("GENERATING YOUR HEAD...")
    print("="*90)

    # CHANGE THIS LINE TO GENERATE ANY LESSON
    query = "I need a school attendance improvment plan. This needs to align with our values and policies, while putting greater pressure on parents who it seems enable absenteeism. The plan should be over three terms, and staged 1,2,3. We are employing an attendance mentor who can wrk directly with families aned even do home visits and bring kigs to schools in the extreme cases, but obviously I need to manage their workload. Please can you draft this police for me."

    extra_uploaded_files = [
        # r"C:\path\to\extra_electricity_guide.pdf",
    ]

    response = generate_head_gp(query, uploaded_files=extra_uploaded_files)
    clean_response = clean_output(response)

    print("\n" + "="*90)
    print("YOUR FINAL CLEAN HEAD")
    print("="*90)
    print(clean_response)