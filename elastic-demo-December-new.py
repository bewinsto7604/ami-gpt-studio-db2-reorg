import openai
import pandas as pd
import os
from pandas import json_normalize 
import time
import requests
import json
import streamlit as st
from elasticsearch import Elasticsearch
import streamlit as st
import urllib.parse
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

api_key = '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'
f = open('C:\Ben\December-Release\zcv1-171494-dba-mapping.json')

loader = TextLoader("NGTZ-Errors.txt")
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)
docsearch = FAISS.from_documents(docs, embeddings)
docsearch.save_local("faiss_ngtz_index")

history = StreamlitChatMessageHistory(key="chat_messages")

history.add_user_message("hi!")
history.add_ai_message("whats up?")

llm=OpenAI(temperature=0, openai_api_key=api_key)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
# if len(msgs.messages) == 0:
#   msgs.add_ai_message("How can I help you?")

template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)



mapping = json.load(f)
mapping_string = json.dumps(mapping)
query = {
    "bool": {
        "must": [
            {
                "range": {
                    "EVENT.beg_tstamp": {
                        "gte": "now-120d/d",
                        "lte": "now/d"
                    }
                }
            },
            {
                "term": {
                    "EVENT.utilname.keyword": "BMC AMI Reorg for Db2"
                }
            },
            {
                "bool": {
                    "must_not": {
                        "term": {
                            "EVENT.cond_code.keyword": "0"
                        }
                    }
                }
            }
        ]
    }
}

query2 = {
    "bool": {
      "must": [
        {
          "match": {
            "EVENT.utilname.keyword": "BMC AMI Reorg for Db2"
          }
        },
        {
          "match": {
            "EVENT.jobname": "DBRRGDP1"
          }
        },        
        {
          "range": {
            "EVENT.cond_code.keyword": {
              "gt": 0
            }
          }
        },
        {
          "range": {
            "EVENT.end_tstamp": {
              "gte": "now-120d"
            }
          }
        }
      ]
    }    
}

def es_connect(cid, user, passwd):
    print('hello')
    es = Elasticsearch(cloud_id=cid, basic_auth=(user, passwd))
    return es

def search(querytext):
    source = '"_source": ["EVENT.jobname", "EVENT.utilname", "EVENT.error_reason_cd", "EVENT.error_reason", "EVENT.beg_tstamp"]'
    cid = os.environ['cloud_id']
    cp = os.environ['cloud_pass']
    cu = os.environ['cloud_user']
    es = es_connect(cid, cu, cp)
    index = 'zcv1-171494-dba'
    resp = es.search(index=index,
                     query=query,
                     size=10000,
                     _source=["EVENT.jobname", "EVENT.utilname", "EVENT.error_reason_cd", "EVENT.error_reason", "EVENT.beg_tstamp"])
    # print(resp)
    return resp

def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))

def search2(querytext):
    source = '"_source": ["EVENT.jobname", "EVENT.utilname", "EVENT.error_reason_cd", "EVENT.error_reason", "EVENT.beg_tstamp", "zadv.SYSNAME", "zadv.SYSORIGIN", "zadv.SUBSYSID", "zadv.JOBNAME", "zadv.STEPNAME", "zadv.PROCNAME", "zadv.JOBID", "zadv.PROGRAMNAME"]'
    cid = os.environ['cloud_id']
    cp = os.environ['cloud_pass']
    cu = os.environ['cloud_user']
    es = es_connect(cid, cu, cp)
    index = 'zcv1-171494-dba'
    resp = es.search(index=index,
                     query=query2,
                     size=10000,
                     _source=["zadv.SYSNAME", "zadv.SYSORIGIN", "zadv.SUBSYSID", "zadv.STEPNAME", "zadv.PROCNAME", "zadv.JOBID", "zadv.PROGRAMNAME"])
    # print(resp)
    json_data = json.dumps(resp['hits']['hits'])
    json_dict = json.loads(json_data)
    return str(json_data)

def search_result2(i: int, SYSNAME: str, SYSORIGIN: str, SUBSYSID: str, STEPNAME: str, PROCNAME: str,
                   JOBID: str, PROGRAMNAME: str,
                  **kwargs) -> str:
    df = pd.DataFrame(
        [
            {"Job Name": "DBRRGDP1", "Util Name": "BMC AMI Reorg for Db2", "Error Reason": "ERROR OCCURRED IN STATEM", "Error Reason Code": "NGTZ170", "SYSNAME": SYSNAME}
        ]
    )

    st.dataframe(df, use_container_width=True)

def search_result(i: int, jobname: str, utilname: str, error_reason: str, error_reason_cd: str, beg_tstamp: str,
                  **kwargs) -> str:
    """ HTML scripts to display search results. """
    return f"""
        <div style="font-size:95%;">
            <div style="color:grey;font-size:95%;">
                Jobname: ·&nbsp; {jobname[:90] + '...' if len(jobname) > 100 else jobname}
            </div>        
            <div style="color:grey;font-size:95%;">
                Utilname: ·&nbsp; {utilname[:90] + '...' if len(utilname) > 100 else utilname}
            </div>
            <div style="float:left;font-style:italic;">
                Error Reason: ·&nbsp; {error_reason} ·&nbsp;
            </div>
            <div style="float:left;font-style:italic;">
                Error Reason Code: ·&nbsp; {error_reason_cd} ·&nbsp;
            </div>
            <div style="float:left;font-style:italic;">
                Timestamp: ·&nbsp; {beg_tstamp} ·&nbsp;
            </div>          
        </div>
    """

company_logo = 'https://www.app.nl/wp-content/uploads/2019/01/Blendle.png'
# Configure Streamlit page
st.set_page_config(
    page_title="Your Notion Chatbot",
    page_icon=company_logo
)
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi human! I am BMC AMI Reorg utility assistant for DB2. How can I help you today?"}]
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt, memory=memory)
# Chat logic
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=company_logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        response = ""
        if "Show me all" in prompt:
            print("enter")
            resp = search(query)
            for hit in resp['hits']['hits']:
                result = hit['_source']['EVENT']
                response = search_result(hit, **result)
                # st.write(search_result(hit, **result), unsafe_allow_html=True)            
                # response = search_result(hit, **result)  
                st.write(response, unsafe_allow_html=True)
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response}) 
        elif "Show me details" in prompt:
            print("enter2")
            resp2 = search2(query)    
            df = pd.read_json(resp2)
            # print(df)
            st.write(df, unsafe_allow_html=True)
            #message_placeholder.markdown(df)
            st.session_state.messages.append({"role": "assistant", "content": df})            
            # print (df)
            # for hit in resp2['hits']['hits']:
            #     result = hit['_source']['zadv']
            #     response = search_result2(hit, **result)
            #     st.write(response, unsafe_allow_html=True)
            #     message_placeholder.markdown(response)
            #     st.session_state.messages.append({"role": "assistant", "content": response}) 
            # print(response)                
        else:    
            result = qa.run(prompt)
            response = result   
        # result = chain({"question": query})
        # response = result['answer']
            full_response = ""
        # Simulate stream of response with milliseconds delay
            for chunk in response.split():
                print(chunk)
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response}) 

def get_completion(prompt, model="gpt-3.5-turbo"):

    # messages = [{"role": "system", "content": prompt}]
    messages = [{"role": "system", "content": query1}, {"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(model=model,messages=messages,temperature=0)

    return response.choices[0].message["content"]

def set_session_state():
    # set default values
    if 'search' not in st.session_state:
        st.session_state.search = None
    if 'tags' not in st.session_state:
        st.session_state.tags = None
    if 'page' not in st.session_state:
        st.session_state.page = 1

    # get parameters in url
    para = st.experimental_get_query_params()
    if 'search' in para:
        st.experimental_set_query_params()
        # decode url
        new_search = urllib.parse.unquote(para['search'][0])
        st.session_state.search = new_search
    if 'tags' in para:
        st.experimental_set_query_params()
        st.session_state.tags = para['tags'][0]
    if 'page' in para:
        st.experimental_set_query_params()
        st.session_state.page = int(para['page'][0])

# print(resp)
# System: You are an expert at the Elasticsearch query DSL. In this conversation I will give you an example of a DSL query you need to make, and you will respond with that query. Always include EVENT prefix for the generated query.
# In the prompt you are provided, when cond_code is not equal to 0, it denotes a failed job.
# Context: When cond_code which is not equal to 0 it denotes a failed job. cond_code must be greater than zero. Show me EVENT.jobname, EVENT.utilname, EVENT.error_reason_cd, EVENT.error_reason for all failed jobs for EVENT.utilname BMC AMI Reorg for Db2. Timestamp is EVENT.beg_tstamp.
# User: Show me all failed jobs for BMC AMI Reorg for Db2 in the last 120 days. Always include EVENT prefix for the generated query.

query1 = "You are an expert at the Elasticsearch query DSL. In this conversation I will give you an example of a DSL query you need to make, and you will respond with that query. In the prompt you are provided, when cond_code is not equal to 0, it denotes a failed job. " + "Timestamp is beg_tstamp. " + "Show me jobname, utilname, error_reason_cd, error_reason for all failed jobs for EVENT.utilname BMC AMI Reorg for Db2. "
query2 = "Show me all failed jobs for BMC AMI Reorg for Db2 in the last 120 days. Prefix JSON attrributes with EVENT schema name."
prompt = 'Given the mapping delimited by triple backticks ``` ' + mapping_string + ' ``` translate the text delimited by triple quotes in a valid Elasticsearch DSL query """ ' + query2 + ' """. Give me only the json code part of the answer. Compress the json output removing spaces.'

def main():
    st.set_page_config(
        layout="wide"
    )
    st.title("ElasticDocs GPT")

    # Main chat form
    with st.form("chat_form"):
        query_text = st.text_input("Enter your query: ")
        submit_button = st.form_submit_button("Send")

    # Generate and display response on form submission
    if submit_button:
        # response = get_completion(query)
        resp = search(query)
        for hit in resp['hits']['hits']:
            result = hit['_source']['EVENT']
            st.write(search_result(hit, **result), unsafe_allow_html=True)                    

def tag_boxes(search: str, tags: list, active_tag: str) -> str:
    """ HTML scripts to render tag boxes. """
    html = ''
    search = urllib.parse.quote(search)
    for tag in tags:
        if tag != active_tag:
            html += f"""
            <a id="tags" href="?search={search}&tags={tag}">
                {tag.replace('-', ' ')}
            </a>
            """
        else:
            html += f"""
            <a id="active-tag" href="?search={search}">
                {tag.replace('-', ' ')}
            </a>
            """

    html += '<br><br>'
    return html


# if __name__ == "__main__":
#    main()