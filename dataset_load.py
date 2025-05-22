import os
import json
import os
import json
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from unstructured.partition.pdf import partition_pdf
os.environ["TABLE_IMAGE_CROP_PAD"] = "2"

load_dotenv()

dataset = {}

directory = 'dataset/extracted'
input_dir = 'dataset/raw'
output_dir = 'dataset/extracted'

for filename in os.listdir(input_dir):
    if filename.endswith(".pdf"):
        print(filename)
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)

        elements = partition_pdf(
            input_path,
            strategy="hi_res",
            infer_table_structure=True,
            hi_res_model_name="detectron2_onnx",
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([el.to_dict() for el in elements], f, ensure_ascii=False, indent=2)

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        with open(filepath, encoding='utf-8') as file:
            data = json.load(file)
            text = ''
            for element in data:
                if element['type'] == 'Table':
                    text += element['metadata']['text_as_html'].strip() + ' '
                else:
                    text += element['text'].strip() + ' '

        fname_upper = filename.upper()

        # Extract metadata
        year_match = re.search(r'20\d\d', fname_upper)
        year = year_match.group(0) if year_match else 'Unknown'

        company = 'Unknown'
        if 'META' in fname_upper:
            company = 'Meta'
        elif 'MICROSOFT' in fname_upper:
            company = 'Microsoft'
        elif 'NVIDIA' in fname_upper:
            company = 'Nvidia'
        elif 'APPLE' in fname_upper:
            company = 'Apple'
        elif 'GOOGLE' in fname_upper or 'ALPHABET' in fname_upper:
            company = 'Google'

        doc_type = 'Unknown'
        if '10Q' in fname_upper:
            doc_type = '10Q'
        elif '10K' in fname_upper or '10-K' in fname_upper:
            doc_type = '10K'
        elif 'ANNUAL' in fname_upper:
            doc_type = 'Annual Report'

        if doc_type == '10Q':
            quarter_match = re.search(r'[1-4]Q', fname_upper)
            quarter = quarter_match.group(0) if quarter_match else 'Unknown'
        else:
            quarter = 'All'

        dataset[filename[:-5]] = {
            'text': text.strip(),
            'year': year,
            'company': company,
            'type': doc_type,
            'quarter': quarter
        }

docs = []
for document in dataset:
    docs.append(Document(
        page_content=dataset[document]['text'],
        metadata={'source':document, 
                  'year': dataset[document]['year'], 
                  'company': dataset[document]['company'],
                  'type': dataset[document]['type'],
                  'quarter': dataset[document]['quarter']
        }
    ))

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv('GEMINI_API_KEY_1'))
text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=85.0)

docs = []
for document in dataset:
    text_chunks = text_splitter.split_text(dataset[document]['text'])
    for text_chunk in text_chunks:
        docs.append(Document(
            page_content=text_chunk,
            metadata={'source':document, 
                    'year': dataset[document]['year'], 
                    'company': dataset[document]['company'],
                    'type': dataset[document]['type'],
                    'quarter': dataset[document]['quarter']}))


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv('GOOGLE_API_KEY'))
persistent_client = chromadb.PersistentClient(path="chroma/")
collection = persistent_client.get_or_create_collection("big_tech_financial_reports")

vector_store = Chroma(
    client=persistent_client,
    collection_name="big_tech_financial_reports",
    embedding_function=embeddings,
)

vector_store._collection.get()