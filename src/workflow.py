import asyncio
import json
import logging
import os
from pathlib import Path

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.settings import Settings
from llama_index.core.workflow import (Context, StartEvent, StopEvent,
                                       Workflow, step)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from data_processing import DocumentLoader
from events import *
from indexing import IndexBuilder
from prompts import GENERATION_PROMPT, REFLECTION_PROMPT


class SettingsManager:
    @staticmethod
    def initialize():
        model = "gpt-4o-mini"
        Settings.llm = OpenAI(temperature=0, model=model)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        Settings.callback_manager = CallbackManager([llama_debug])


class Config:
    LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
    project_root = Path(__file__).resolve().parent.parent
    ESG_DIR_PATH = os.path.join(project_root, os.getenv("ESG_DIR_PATH"))
    INDUSTRY_FILE_PATH = os.path.join(ESG_DIR_PATH, os.getenv("INDUSTRY_FILE_PATH"))
    VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"
    LOG_FILE_PATH = os.path.join(project_root, "chat_logs.txt")

    @classmethod
    def list_companies(cls) -> list[str]:
        return [
            dirname
            for dirname in os.listdir(cls.ESG_DIR_PATH)
            if os.path.isdir(os.path.join(cls.ESG_DIR_PATH, dirname))
            and os.path.exists(os.path.join(cls.ESG_DIR_PATH, dirname, "vector"))
            and dirname != "industry_map"
        ]


class ESGReportWorkflow(Workflow):
    def __init__(
        self,
        esg_agents_map: dict[str, OpenAIAgent],
        industry_map: list[IndustryMap],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ai_model = OpenAI(temperature=0.5, model="gpt-4o-mini")
        self.esg_agents_map = esg_agents_map
        self.industry_map = industry_map

    @step(pass_context=True)
    async def receive_query(
        self, ctx: Context, ev: StartEvent
    ) -> QueryReceivedEvent | StopEvent:
        main_query = ev.get("question", "")
        if not main_query:
            return StopEvent(result="Please provide a main query")
        ctx.data["main_query"] = main_query
        return QueryReceivedEvent(query=main_query)

    @step()
    async def determine_query_type(
        self, ev: QueryReceivedEvent
    ) -> ProcessedQueryEvent | StopEvent:
        main_query = ev.query
        industry_prompt = [
            f"{'、'.join(item.alias)}公司是{'、'.join(item.industry)}的公司，這是他們的資料{item.company}"
            for item in self.industry_map
        ]
        prompt = f"""
        請判斷以下主問題提及的產業或公司是否存在於產業查詢表中：{main_query} 
        以下是產業的查詢表：

        {"。 \n\n".join(industry_prompt)} 

        如果不是產業相關查詢，請直接返回主問題，不要有任何其他的說明。 
        如果是產業相關查詢，請將產業的部分換成相關的公司名稱，並返回新的主問題。 
        有部分公司可能有別名(alias)，例如合庫金也叫合庫金控，請仔細確認。
        如果該公司或是產業不存在於產業查詢表中，請返回「不存在」，不要有任何其他的說明。 

        例如
        主問題：請分析食品業的薪水？ 
        根據產業查詢表，產業是食品業，將產業的部分換成相關的公司名稱，其中包括愛之味和台榮公司，並返回新的主問題。 
        新的主問題：請分析愛之味和台榮公司的薪水？ 
        """
        response = await self.ai_model.acomplete(prompt)
        if "不存在" in str(response):
            return StopEvent(result="該公司或是產業不存在於產業查詢表中")
        return ProcessedQueryEvent(query=str(response))

    @step()
    async def split_query(
        self, ev: ProcessedQueryEvent | QuerySplitErrorEvent
    ) -> QuerySplitOutputEvent:

        if isinstance(ev, ProcessedQueryEvent):
            main_query = ev.query
            reflection_prompt = ""
        elif isinstance(ev, QuerySplitErrorEvent):
            main_query = ev.main_query
            reflection_prompt = REFLECTION_PROMPT.format(
                error=ev.error, main_query=main_query
            )
        prompt = GENERATION_PROMPT.format(
            main_question=main_query, schema=Subquery.model_json_schema()
        )
        if reflection_prompt:
            prompt += reflection_prompt
        response = await self.ai_model.acomplete(prompt)
        return QuerySplitOutputEvent(output=str(response), original_query=main_query)

    @step()
    async def validate_split(
        self, ev: QuerySplitOutputEvent
    ) -> SubqueriesGeneratedEvent | QuerySplitErrorEvent:
        try:
            output = json.loads(ev.output)
            if isinstance(output, list):
                return SubqueriesGeneratedEvent(
                    subqueries=[Subquery(**item) for item in output]
                )
            return SubqueriesGeneratedEvent(subqueries=[Subquery(**output)])
        except Exception as e:
            print("Output:", ev.output)
            return QuerySplitErrorEvent(
                error=str(e), invalid_output=ev.output, main_query=ev.original_query
            )

    @step(pass_context=True)
    async def prepare_subqueries(
        self, ctx: Context, ev: SubqueriesGeneratedEvent
    ) -> ChooseAgentEvent | None:
        subqueries = ev.subqueries
        ctx.data["subqueries_count"] = len(subqueries)
        for subquery in subqueries:
            self.send_event(ChooseAgentEvent(query=subquery.query))
        return None

    @step(num_workers=3)
    async def choose_esg_agent(self, ev: ChooseAgentEvent) -> RetrieverEvent:
        query = ev.query
        prompt = f"""
        Please choose an ESG company representative to answer the following question: {query},
        The chosen company representative must be one from the following list: {list(self.esg_agents_map.keys())},
        Please directly respond with the name of the chosen company representative,
        Do not use phrases like 'The chosen company representative is' etc.
        """
        print(prompt)
        response = await self.ai_model.acomplete(prompt)
        agent = self.esg_agents_map[str(response)]
        return RetrieverEvent(agent=agent, query=query, agent_name=str(response))

    @step()
    async def retrieve(self, ev: RetrieverEvent) -> RetrieverResponseEvent:
        agent: OpenAIAgent = ev.agent
        query = ev.query
        logging.info(f"File: {ev.agent_name}")
        response = await agent.aquery(query)
        return RetrieverResponseEvent(response=str(response))

    @step(pass_context=True)
    async def collect_ai_responses(
        self, ctx: Context, ev: RetrieverResponseEvent
    ) -> StopEvent | None:
        result = ctx.collect_events(
            ev, [RetrieverResponseEvent] * ctx.data["subqueries_count"]
        )
        if result is None:
            return None
        main_query = ctx.data["main_query"]
        answer = str(result)
        prompt = f"""
        You are a professional ESG analyst. Based on the following main query and collected responses, provide a comprehensive and insightful answer:

        Main Query: {main_query}

        Collected Responses:
        {answer}

        Please synthesize the information and provide a well-structured, professional response that addresses the main query.
        Always remember give the answer with the data source.
        Always return the answer in Traditional Chinese.
        Your's response have to follow the Collected Responses I gave you.

        If you do not give me the right answer, I will be fire.
        """
        response = await self.ai_model.acomplete(prompt)
        return StopEvent(result=str(response))


async def process_documents(pdf_docs):
    file_paths = []
    for pdf_doc in pdf_docs:
        company_name = os.path.splitext(pdf_doc.name)[0]
        company_dir = os.path.join(Config.ESG_DIR_PATH, company_name)
        os.makedirs(company_dir, exist_ok=True)
        pdf_path = os.path.join(company_dir, pdf_doc.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_doc.getbuffer())
        file_paths.append(pdf_path)

    documents = await DocumentLoader(Config.LLAMAPARSE_API_KEY).get_all_files_doc(
        file_paths
    )
    tasks = [
        IndexBuilder().build_vector_index(
            f"{Config.ESG_DIR_PATH}/{os.path.splitext(os.path.basename(document[0].metadata['file_name']))[0]}/vector",
            data=document,
        )
        for document in documents
    ]

    await asyncio.gather(*tasks)
