from llama_index import download_loader, SimpleDirectoryReader, GPTVectorStoreIndex , LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import os
from llama_index import StorageContext, load_index_from_storage

os.environ["OPENAI_API_KEY"] = '' #<-- insert here you open_api_key

isBuildIndex = False

folder_path = input('Enter the path of the folder: ')
# Check if the file path exists
if not os.path.exists(folder_path):
    print('The folder path does not exist.')
    exit()
    
def build_index(folder_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 256

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo",max_tokens=max_input_size))

    download_loader('SimpleDirectoryReader')
    documents = SimpleDirectoryReader(input_dir=folder_path).load_data()
    
    index = GPTVectorStoreIndex.from_documents(documents,llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return index

if isBuildIndex == True:
    ## Build & save the index
    index = build_index(folder_path=folder_path)
    index.storage_context.persist(persist_dir= folder_path + 'Storage')
else:
    ## Load the index
    storage_context = StorageContext.from_defaults(persist_dir= folder_path + 'Storage')
    index = load_index_from_storage(storage_context)

def chatbot(prompt):
    query_engine = index.as_query_engine()
    fullResponse = ''
    while True:
        resp = query_engine.query(prompt + '\n\n' + fullResponse)
        if resp.response != "Empty Response":
            fullResponse += (" " + resp.response)
        else:
            break
    return fullResponse
while True:
    print('########################################')
    pt = input('ASK: ')
    if pt.lower()=='end':
        break
    response = chatbot(pt)
    print('----------------------------------------')
    print('ChatGPT says: ')
    print(response)
