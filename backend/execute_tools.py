'''
Выполнение инструментов (SQL-запросы) и логика переходов между нодами
'''

import logging
import json

from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage


from agent.state import AgentState
from database import run_sql, get_postgres_schema

logger = logging.getLogger(__name__)

sql_tool_node = ToolNode(tools=[run_sql, get_postgres_schema])

def execute_tool_node(state: AgentState):
    '''Выполняет инструменты и извлекает результаты'''
    
    logger.info("🔧 Выполнение инструмента...")
    tool_result = sql_tool_node.invoke(state)
    last_tool_msg = tool_result["messages"][-1]

    last_sql = state.get("last_sql")
    for m in reversed(state.get("messages", [])):
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if tc.get("name") != "run_sql":
                continue
            args = tc.get("args") or {}
            if isinstance(args, dict) and isinstance(args.get("query"), str):
                last_sql = args["query"]
                break
        if last_sql and last_sql != state.get("last_sql"):
            break

    new_state = {
        "messages": tool_result["messages"],
        "original_query": state.get("original_query") or "",
        "from_cache": state.get("from_cache", False),
        "visualization_code": state.get("visualization_code"),
        "requires_graph_vis": state.get("requires_graph_vis", False),
        "last_sql": last_sql,
        "data": state.get("data"),
        "query_result": state.get("query_result"),
        "critic_attempts": state.get("critic_attempts", 0),
    }

    tool_name = getattr(last_tool_msg, "name", "")
    
    if tool_name == "get_postgres_schema":
        logger.info("📋 Получена схема БД")
        return new_state
    
    content = last_tool_msg.content.strip()
    if tool_name == "run_sql":
        try:
            result = json.loads(content)
            if not result.get("success", False):
                new_state["sql_error_count"] = state.get("sql_error_count", 0) + 1
                new_state["last_error_message"] = result.get("error", content[:200])
            else:
                new_state["sql_error_count"] = 0
                new_state["last_error_message"] = ""
                data = result.get("data", [])
                new_state["query_result"] = data
                new_state["data"] = data  # Для graph_vis
                logger.info(f"💾 Сохранено {len(data)} строк в query_result")
        except (json.JSONDecodeError, ValueError, TypeError):
            new_state["sql_error_count"] = state.get("sql_error_count", 0) + 1
            new_state["last_error_message"] = content[:200]

    return new_state


def should_continue(state: AgentState) -> str:
    """
    Определяет следующий шаг после ноды assistant.
    Используется ТОЛЬКО для переходов из assistant, НЕ из tools.
    """
    logger.info("🔀 === SHOULD_CONTINUE (from assistant) ===")
    
    if state.get("from_cache", False):
        logger.info("✅ Ответ из кэша → END")
        return END
    
    if state.get("sql_error_count", 0) >= 4:
        logger.warning(f"❌ Слишком много ошибок SQL ({state['sql_error_count']}) → END")
        return END
    
    last_msg = state["messages"][-1]
    msg_type = type(last_msg).__name__
    msg_name = getattr(last_msg, 'name', 'N/A')
    has_tool_calls = hasattr(last_msg, 'tool_calls') and bool(last_msg.tool_calls)
    
    logger.info(f"📝 Последнее сообщение: type={msg_type}, name={msg_name}, has_tool_calls={has_tool_calls}")
    
    if has_tool_calls:
        logger.info("✅ Есть tool_calls → tools")
        return "tools"
    
    if isinstance(last_msg, AIMessage):
        if msg_name != "sql_critic":
            logger.info("ℹ️ Финальный ответ ассистента → END")
            return END
    
    return END