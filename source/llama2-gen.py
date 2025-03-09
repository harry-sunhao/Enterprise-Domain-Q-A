from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
import torch
import transformers
from transformers import AutoModel
from langchain.prompts import PromptTemplate
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from huggingface_hub import login

class CustomEmbeddings:
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to("cuda")

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def setup_qa_system(llm, document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    texts = text_splitter.split_documents([document])
    
    embeddings = CustomEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)
    retriever = vectordb.as_retriever()

    prompt_template = """
Context: {context}

Question: {question}

Please provide a concise answer based on the context above.
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=False
    )

def setup_model():
    login(token="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")
    #if the token does not work, get an new token from huggingface with llama2 access authorized
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return HuggingFacePipeline(pipeline=pipeline)

def load_documents(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_qa_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_bleu_score(reference, candidate):
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    return sentence_bleu([reference_tokens], candidate_tokens)

def clean_answer(answer):
    # Remove any prefixes like "Answer:" or "Context:"
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer

def main():
    # Load documents and QA data
    # this part of program is used in colab to utilize GPU, change the path if running on other platform
    try:
        documents = load_documents('../data/training_dev_technotes.json')
        qa_data = load_qa_data('../data/training_Q_A.json')
    except FileNotFoundError as e:
        print(f"Error: Could not load input files - {e}")
        return
    
    # Setup model
    llm = setup_model()
    
    # Store results
    results = []
    total_bleu = 0
    processed_questions = 0
    
    # Process each question
    for qa_item in qa_data:
        if qa_item['ANSWERABLE'] == 'Y':
            doc_id = qa_item['DOCUMENT']
            if doc_id not in documents:
                print(f"Document {doc_id} not found")
                continue
                
            doc_content = documents[doc_id]
            document = Document(
                page_content=f"{doc_content['title']}\n\n{doc_content['text']}",
                metadata={'source': doc_id}
            )
            
            qa = setup_qa_system(llm, document)
            question = f"{qa_item['QUESTION_TITLE']}\n{qa_item['QUESTION_TEXT']}"
             # Get and clean answer
            raw_answer = qa.run(question)
            answer = clean_answer(raw_answer)

            bleu_score = calculate_bleu_score(qa_item['ANSWER'], answer)
            
            # Store result
            result = {
                'question_id': qa_item['QUESTION_ID'],
                'doc_id': doc_id,
                'question': question,
                'predicted_answer': answer,
                'ground_truth': qa_item['ANSWER'],
                'bleu_score': bleu_score
            }
            results.append(result)
            
            # Print minimal output format
            
            #print(f"{qa_item['QUESTION_ID']} Answer: {raw_answer}")
            print(f"{qa_item['QUESTION_ID']} Answer: {answer}")
            print(f"Ground Truth: {qa_item['ANSWER']}")
            print(f"BLEU Score: {bleu_score}\n")
            
            total_bleu += bleu_score
            processed_questions += 1
    
    if processed_questions > 0:
        print(f"Average BLEU Score: {total_bleu/processed_questions:.4f}")
    
    # Save results to file
    try:
        with open('output/qa_train_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    except IOError as e:
        print(f"Error saving results file: {e}")

if __name__ == "__main__":
    main()
