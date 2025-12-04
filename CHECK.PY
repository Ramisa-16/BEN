from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_4L26ra_HZYXdtHKRrRuTAyEjpEzherH6AkpEB7bJXWpkJkfEDfJbqMUdbhs7vEpsrE3181")
index = pc.Index("lesson-gem-index")

# List of your 14 PDF filenames (exactly as they appear in the path)
expected_files = [
    "Autism primary schools.pdf",
    "Behaviour Improving in Primary schools EEF.pdf",
    "EAL Support strategies Primary Bell Foundation.pdf",
    "Literacy improvement in KS2 report Second edition EEF.pdf",
    "Literacy improvements in KS1 Guidance Report 2020 EEF.pdf",
    "Maths Early Years to KS1 Guidance Report EEF.pdf",
    "National_curriculum Primary (1).pdf",
    "Maths KS2 KS3 improvements 2022 Guidence report Update EEF.pdf",
    "Primary policy book.docx.pdf",
    "Science improvement in primary guidance-report-ks1-ks2 EEF.pdf",
    "SEMH - Primary school Social and Emotional Learning EEF.pdf",
    "Teacher assessment frameworks KS2.pdf",
    "Teacher assessment frameworks- Non stat - GOV.UK.pdf",
    "TRAUMA-INFORMED INSTRUCTION - Hanover institute.pdf"
]

# Check which files actually exist in Pinecone
found = set()
missing = []

for file in expected_files:
    # Search for just ONE vector that has this source
    results = index.query(
        vector=[0.0] * 768,  # dummy vector
        top_k=1,
        filter={"source": {"$eq": file}},
        include_metadata=True
    )
    if results.matches:
        found.add(file)
    else:
        missing.append(file)

print("ALL PDFs IN PINECONE?")
print("="*60)
print(f"Found: {len(found)} / 14")
print(f"Missing: {len(missing)}")
print("="*60)

if missing:
    print("MISSING FILES (will not be used by AI):")
    for f in missing:
        print(f"   → {f}")
else:
    print("PERFECT! ALL 14 PDFs are fully uploaded and active!")

print("\nSources found:")
for f in sorted(found):
    print(f"   {f}")