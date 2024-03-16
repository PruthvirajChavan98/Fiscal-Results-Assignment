from IndexManager.IndexManager import IndexManager, Document, VectorStoreIndex
import streamlit as st

# Initialize the comprehensive index manager
manager = IndexManager(
    engine="<YOUR DEPLOYMENT NAME>",
    model="gpt-35-turbo-16k",
    api_key="<API KEY>",
    endpoint="AZURE ENDPOINT",
    version="2024-02-15-preview",
    streaming=False,
    max_tokens=1024,
    embed_model_name="BAAI/bge-small-en-v1.5",
)

# Infosys 2020-2021
infosys_20_21_index_dir = "./index_storage/infosys-20-21"
infosys_20_21_docs = ["./Assignment_dataset/infosys-20-21.pdf"]
infosys_20_21_index = manager.load_or_build_index(infosys_20_21_index_dir, infosys_20_21_docs)

# Infosys 2021-2022
infosys_21_22_index_dir = "./index_storage/infosys-21-22"
infosys_21_22_docs = ["./Assignment_dataset/infosys-21-22.pdf"]
infosys_21_22_index = manager.load_or_build_index(infosys_21_22_index_dir, infosys_21_22_docs)

# Infosys 2022-2023
infosys_22_23_index_dir = "./index_storage/infosys-22-23"
infosys_22_23_docs = ["./Assignment_dataset/infosys-22-23.pdf"]
infosys_22_23_index = manager.load_or_build_index(infosys_22_23_index_dir, infosys_22_23_docs)


# TCS 2020-2021
tcs_20_21_index_dir = "./index_storage/tcs-20-21"
tcs_20_21_docs = ["./Assignment_dataset/tcs-20-21.pdf"]
tcs_20_21_index = manager.load_or_build_index(tcs_20_21_index_dir, tcs_20_21_docs)

# TCS 2021-2022
tcs_21_22_index_dir = "./index_storage/tcs-21-22"
tcs_21_22_docs = ["./Assignment_dataset/tcs-21-22.pdf"]
tcs_21_22_index = manager.load_or_build_index(tcs_21_22_index_dir, tcs_21_22_docs)

# TCS 2022-2023
tcs_22_23_index_dir = "./index_storage/tcs-22-23"
tcs_22_23_docs = ["./Assignment_dataset/tcs-22-23.pdf"]
tcs_22_23_index = manager.load_or_build_index(tcs_22_23_index_dir, tcs_22_23_docs)


# Wipro 2020-2021
wipro_20_21_index_dir = "./index_storage/wipro-20-21"
wipro_20_21_docs = ["./Assignment_dataset/wipro-20-21.pdf"]
wipro_20_21_index = manager.load_or_build_index(wipro_20_21_index_dir, wipro_20_21_docs)

# Wipro 2021-2022
wipro_21_22_index_dir = "./index_storage/wipro-21-22"
wipro_21_22_docs = ["./Assignment_dataset/wipro-21-22.pdf"]
wipro_21_22_index = manager.load_or_build_index(wipro_21_22_index_dir, wipro_21_22_docs)

# Wipro 2022-2023
wipro_22_23_index_dir = "./index_storage/wipro-22-23"
wipro_22_23_docs = ["./Assignment_dataset/wipro-22-23.pdf"]
wipro_22_23_index = manager.load_or_build_index(wipro_22_23_index_dir, wipro_22_23_docs)



infosys_20_21_tool = manager.setup_tool(
    index=infosys_20_21_index,
    name="infosys_20_21_reports_retriever_tool",
    description="Provides information about Infosys annual report for the fiscal year 2020-2021. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)

infosys_21_22_tool = manager.setup_tool(
    index=infosys_21_22_index,
    name="infosys_21_22_reports_retriever_tool",
    description="Provides information about Infosys annual report for the fiscal year 2021-2022. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)

infosys_22_23_tool = manager.setup_tool(
    index=infosys_22_23_index,
    name="infosys_22_23_reports_retriever_tool",
    description="Provides information about Infosys annual report for the fiscal year 2022-2023. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)


tcs_20_21_tool = manager.setup_tool(
    index=tcs_20_21_index,
    name="tcs_20_21_reports_retriever_tool",
    description="Provides information about TCS annual report for the fiscal year 2020-2021. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)

tcs_21_22_tool = manager.setup_tool(
    index=tcs_21_22_index,
    name="tcs_21_22_reports_retriever_tool",
    description="Provides information about TCS annual report for the fiscal year 2021-2022. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)

tcs_22_23_tool = manager.setup_tool(
    index=tcs_22_23_index,
    name="tcs_22_23_reports_retriever_tool",
    description="Provides information about TCS annual report for the fiscal year 2022-2023. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)


wipro_20_21_tool = manager.setup_tool(
    index=wipro_20_21_index,
    name="wipro_20_21_reports_retriever_tool",
    description="Provides information about Wipro annual report for the fiscal year 2020-2021. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)

wipro_21_22_tool = manager.setup_tool(
    index=wipro_21_22_index,
    name="wipro_21_22_reports_retriever_tool",
    description="Provides information about Wipro annual report for the fiscal year 2021-2022. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)

wipro_22_23_tool = manager.setup_tool(
    index=wipro_22_23_index,
    name="wipro_22_23_reports_retriever_tool",
    description="Provides information about Wipro annual report for the fiscal year 2022-2023. \nUse a detailed plain text question as input to the tool.",
    tool_type="query"
)


texts = [
    "Abbreviation: KRD = Key Revenue Driver",
    "Abbreviation: CM = Chairman's Message",
    "Abbreviation: RF = Risk Factors",
    "Abbreviation: YR = Year",
    "Abbreviation: REV = Revenue",
    "Abbreviation: RIS = Risks",
    "Abbreviation: COMP = Comparison",
]
docs = [Document(text=t) for t in texts]
context_index = VectorStoreIndex.from_documents(docs, service_context=manager.service_context)


context_agent = manager.create_context_agent(
    tools=[
        infosys_20_21_tool,
        infosys_21_22_tool,
        infosys_22_23_tool,
        tcs_20_21_tool,
        tcs_21_22_tool,
        tcs_22_23_tool,
        wipro_20_21_tool,
        wipro_21_22_tool,
        wipro_22_23_tool
    ],
    context_index=context_index,  # This needs to be an already initialized VectorStoreIndex for context
    system_prompt="""
    Efficiently process and respond to multiple questions in a single input, 
    accurately identifying the intent of each question. 
    Utilize appropriate inputs and tools as necessary to provide comprehensive answers.
    """
)

st.title("Annual Reports Bot")

user_input = st.text_input("Feel free to ask me anything about Infosys, Wipro, or TCS annual reports from 2020 to 2023!", "")

if user_input:
    # Assuming `context_agent` is already initialized and ready to use as per your provided setup
    response = str(context_agent.chat(user_input))
    st.text_area("Response:", value=response, height=300)