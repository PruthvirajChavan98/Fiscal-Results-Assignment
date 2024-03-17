from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext, Document, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata, RetrieverTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.agent.openai_legacy import ContextRetrieverOpenAIAgent
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
import os

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
        elif tool_type == "automerging_query_engine":
            tool = QueryEngineTool(
                query_engine=self.get_automerging_query_engine(index),
                metadata=ToolMetadata(name=name, description=description)
            )
        
        else:
            raise ValueError("Invalid tool type specified. Choose 'query' or 'retriever' or 'automerging_query_engine'.")
        return tool

    def create_context_agent(self, tools, context_index: VectorStoreIndex, system_prompt):
        context_retriever = context_index.as_retriever(similarity_top_k=1)  # Assuming context_index is provided
        context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
            tools=tools,
            retriever=context_retriever,
            verbose=True,
            llm=self.llm,
            system_prompt=system_prompt
        )
        return context_agent
    
    def build_or_load_automerging_index(self, persist_dir, input_files, chunk_sizes=None):
        # Set default chunk sizes if not provided
        chunk_sizes = chunk_sizes or [2048, 512, 128]
        
        # Initialize node parser with default chunk sizes
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        
        # Check if the index already exists and load it if possible
        if os.path.exists(persist_dir):
            print(f"Loading existing index from {persist_dir}")
            try:
                return load_index_from_storage(
                    StorageContext.from_defaults(persist_dir=persist_dir),
                    service_context=self.service_context
                )
            except Exception as e:
                print(f"Error loading existing index from {persist_dir}: {e}")

        # If index does not exist or failed to load, proceed to build a new one
        print(f"Building new index at {persist_dir}")
        documents = SimpleDirectoryReader(input_files=input_files).load_data()
        doc_text = "\n\n".join([d.get_content() for d in documents])
        docs = [Document(text=doc_text)]
        
        # Process documents into nodes without concatenating texts unnecessarily
        nodes = node_parser.get_nodes_from_documents(docs)
        leaf_nodes = get_leaf_nodes(nodes)
        
        # Prepare storage context
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        # Create and persist the automerging index
        try:
            automerging_index = VectorStoreIndex(
                leaf_nodes, storage_context=storage_context, service_context=self.service_context, show_progress=True
            )
            automerging_index.storage_context.persist(persist_dir=persist_dir)
            return automerging_index
        except Exception as e:
            print(f"Failed to build or persist new index at {persist_dir}: {e}")
            raise

    def get_automerging_query_engine(self, automerging_index: VectorStoreIndex, similarity_top_k=12, rerank_top_n=6):
        base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = AutoMergingRetriever(
            vector_retriever=base_retriever, storage_context=automerging_index.storage_context, verbose=True
        )

        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )

        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, service_context=self.service_context, node_postprocessors=[rerank]
        )
        return auto_merging_engine