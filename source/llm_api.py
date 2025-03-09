import json
from openai import OpenAI


deepseek_cfg = {
    "token": "sk-edb1d3602d1d485a88ab517fa1ac1a67",
    "url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
}

cfg = deepseek_cfg

prompt_from_question_to_keywords = """
Task: Information Extraction. 
Please read the document carefully and extract information to answer the questions based on the given content. 
Context: {context}
REMEMBER:
- Only output exactly the same sentences from the context without providing any inference, no longer than 200 words.
Question: {question}
"""


def get_model(cfg):
    token = cfg['token']
    client = OpenAI(api_key=token, base_url=cfg['url'])
    return client


def llm_generate(user_content, client, cfg):
    messages = [
        {
            "role": "user",
            "content": user_content,
        },
    ]

    completion = client.chat.completions.create(
        model=cfg['model'],
        messages=messages,
        temperature=0.,
    )
    return completion.choices[0].message.content


if __name__ == '__main__':

    client = get_model(cfg)

    documents_path = '../data/training_dev_technotes.json'
    with open(documents_path, 'r') as d_file:
        documents = json.load(d_file)

    qa_data_path = '../data/dev_Q_A.json'
    with open(qa_data_path, 'r') as file:
        qa_data = json.load(file)

    outputs = []

    i = 1
    for qa_item in qa_data:
        print(i)
        i += 1
        if qa_item['ANSWERABLE'] == 'N':
            outputs.append('')
        else:
            doc_id = qa_item['DOCUMENT']
            if doc_id not in documents:
                print(f"Document {doc_id} not found")
                continue

            doc_content = documents[doc_id]
            page_content = f"{doc_content['title']}\n{doc_content['text']}",

            question = f"{qa_item['QUESTION_TITLE']}\n{qa_item['QUESTION_TEXT']}"

            user_content = prompt_from_question_to_keywords.format(context=page_content, question=question)

            response = llm_generate(user_content=user_content, client=client, cfg=cfg)

            outputs.append(question)

    print(len(outputs))

    csv_file_path = 'dev.out'
    with open(csv_file_path, "w") as f:
        for item in outputs:
            key = {"output": item}
            # print(key)
            f.write(json.dumps(key) + "\n")

    print(f"Keys have been saved to {csv_file_path}")
