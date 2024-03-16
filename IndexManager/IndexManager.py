from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext, Document, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata, RetrieverTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.agent.openai_legacy import ContextRetrieverOpenAIAgent

class IndexManager:
    def __init__(self, engine, model, api_key, endpoint, version, streaming, max_tokens, embed_model_name):
        # Initialize LLM Service
        self.llm = AzureOpenAI(
            engine=engine, 
            model=model, 
            api_key=api_key,
            azure_endpoint=endpoint, 
            api_version=version,
            streaming=streaming, 
            max_tokens=max_tokens,
            temperature=0.0
            )
        # Initialize Embedding Service
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        # Setup Service Context
        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)

    def load_or_build_index(self, persist_dir, input_files):
        # Attempt to Load Index
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context, service_context=self.service_context)
            print(f"Index loaded from {persist_dir}.")
        except Exception as e:
            # If load fails, build index
            print(f"Failed to load index from {persist_dir}, building new index. Error: {e}")
            documents = SimpleDirectoryReader(input_files=input_files).load_data()
            index = VectorStoreIndex.from_documents(documents, service_context=self.service_context, show_progress=True)
            index.storage_context.persist(persist_dir=persist_dir)
            print(f"Index built and persisted at {persist_dir}.")
        return index

    def setup_tool(self, index, name, description, tool_type):
        if tool_type == "query":
            tool = QueryEngineTool(
                query_engine=index.as_query_engine(similarity_top_k=3),
                metadata=ToolMetadata(name=name, description=description)
            )
        elif tool_type == "retriever":
            tool = RetrieverTool(
                retriever=index.as_retriever(similarity_top_k=3),
                metadata=ToolMetadata(name=name, description=description)
            )
        else:
            raise ValueError("Invalid tool type specified. Choose 'query' or 'retriever'.")
        return tool

    def create_context_agent(self, tools, context_index, system_prompt):
        context_retriever = context_index.as_retriever(similarity_top_k=1)  # Assuming context_index is provided
        context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
            tools=tools,
            retriever=context_retriever,
            verbose=True,
            llm=self.llm,
            system_prompt=system_prompt
        )
        return context_agent
