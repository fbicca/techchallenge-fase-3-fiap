"""Fábrica do agente SQL (LangChain) — usado por chains legado e LangGraph."""
from __future__ import annotations

from typing import Any

from langchain_community.llms import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase

from .database import ensure_database


def build_sql_agent(llm: HuggingFacePipeline, *, db_path: Any = None):
    path = ensure_database(db_path)
    uri = f"sqlite:///{path}"
    db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=2)
    try:
        from langchain_community.agent_toolkits import create_sql_agent
    except ImportError:  # pragma: no cover
        from langchain_community.agent_toolkits.sql.base import create_sql_agent

    return create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        handle_parsing_errors=True,
    )
