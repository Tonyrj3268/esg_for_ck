import asyncio
import json
import logging
import os
import time

import streamlit as st

from agents import AgentBuilder
from html_template import bot_template, css, user_template
from workflow import (Config, ESGReportWorkflow, IndustryMap, SettingsManager,
                      process_documents)


def initialize_agents():
    SettingsManager.initialize()
    agent_builder = AgentBuilder(Config.ESG_DIR_PATH)
    return agent_builder.build_esg_agents(Config.list_companies())


def update_sidebar_companies() -> None:
    st.sidebar.subheader("公司列表")
    if "companies" not in st.session_state:
        st.session_state.companies = Config.list_companies()
    company_count = len(st.session_state.companies)
    st.sidebar.write(f"目前共有 {company_count} 家企業永續報告書")
    with st.sidebar.expander("點擊展開", expanded=False):
        if st.session_state.companies:
            for company in st.session_state.companies:
                st.write(company)
        else:
            st.write("目前沒有已存的企業永續報告書。")
    if Config.VERBOSE:
        with st.sidebar:
            st.subheader("你的文件")
            pdf_docs = st.file_uploader(
                "在這裡上傳你的PDF並點擊'處理'", accept_multiple_files=True
            )
            if st.button("處理"):
                with st.spinner("處理中"):
                    if pdf_docs:
                        asyncio.run(process_documents(pdf_docs))
                        st.success("文件處理完成！")
                    else:
                        st.write("沒有文件被上傳。")


def handle_userinput(user_question: str) -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    workflow = ESGReportWorkflow(
        timeout=300,
        verbose=True,
        esg_agents_map=st.session_state.esg_agents_map,
        industry_map=st.session_state.industry_map,
    )
    logging.info(f"Question: {user_question}")
    response = asyncio.run(workflow.run(question=user_question))
    logging.info(f"Response: {response}")
    st.session_state.chat_history.append(
        {"role": "assistant", "content": str(response)}
    )
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    for message in st.session_state.chat_history[::-1]:
        template = user_template if message["role"] == "user" else bot_template
        st.write(
            template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True
        )


def main():
    st.set_page_config(page_title="向你的PDF問問題", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("向多個PDF問問題 :books:")
    if "last_activity_time" not in st.session_state:
        st.session_state.last_activity_time = time.time()

    if "esg_agents_map" not in st.session_state:
        st.session_state.esg_agents_map = initialize_agents()
    if "industry_map" not in st.session_state:
        with open(Config.INDUSTRY_FILE_PATH, "r") as f:
            industry_data = json.load(f)
            st.session_state.industry_map = [
                IndustryMap(
                    company=item["company"],
                    industry=item["industry"],
                    alias=item["alias"],
                )
                for item in industry_data
            ]
    user_question = st.text_input(
        "問一個關於你文件的問題：（模型：GPT-4o-mini · 生成的內容可能不准確或錯誤）"
    )

    if user_question:
        handle_userinput(user_question)
    update_sidebar_companies()
