'''
Генерация и выполнение кода визуализации.
'''
import io
import base64
import logging
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from config import GROQ_API_KEY

logger = logging.getLogger(__name__)

llm_graph = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    api_key=GROQ_API_KEY,
    max_tokens=2048,
)


def extract_code(text: str) -> str:
    """Надёжно извлекает чистый Python-код из ответа LLM"""
    import re
    text = text.strip()
    
    code_block = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()
    else:
        if text.startswith("```python"):
            text = text[9:].lstrip()
        elif text.startswith("```"):
            text = text[3:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    
    lines = []
    skip_prefixes = (
        "Код:", "Только код:", "```", "# график", "# визуализация", 
        "//", "/*", "*/", "ответ:", "вот:", "вот"
    )
    for line in text.split("\n"):
        stripped = line.strip().lower()
        if not any(stripped.startswith(prefix.lower()) for prefix in skip_prefixes):
            lines.append(line)
    
    return "\n".join(lines).strip()


def _validate_syntax(code: str) -> str | None:
    """Проверяет синтаксис кода. Возвращает None если OK, или текст ошибки."""
    try:
        compile(code, "<generated>", "exec")
        return None
    except SyntaxError as e:
        return f"Строка {e.lineno}: {e.msg}"


def _render_chart_base64(code: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Выполняет код визуализации и возвращает chart base64 либо текст ошибки."""
    dangerous_patterns = [
        "import os", "import sys", "import subprocess", "import shlex",
        "eval(", "exec(", "__import__", ".system(", ".popen(",
        "os.path", "os.remove", "os.listdir", "os.getcwd", "os.environ",
        "sys.exit", "subprocess.", "shutil.",
        "globals(", "locals(", "__dict__", "__class__",
    ]

    code_lower = code.lower()
    for pattern in dangerous_patterns:
        if pattern in code_lower:
            return None, f"Запрещённая операция: {pattern}"

    syntax_err = _validate_syntax(code)
    if syntax_err:
        return None, f"Синтаксическая ошибка: {syntax_err}"

    safe_globals = {
        "__builtins__": {
            "range": range, "len": len, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
            "enumerate": enumerate, "zip": zip, "sorted": sorted, "reversed": reversed,
            "isinstance": isinstance, "type": type, "hasattr": hasattr, "getattr": getattr,
            "print": print, "abs": abs, "sum": sum, "min": min, "max": max,
            "round": round, "pow": pow, "divmod": divmod, "all": all, "any": any,
            "filter": filter, "map": map,
            "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
            "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
        },
        "pd": pd,
        "np": np,
        "json": json,
        "plt": plt,
        "sns": sns,
    }

    safe_locals = {
        "df": df.copy(),
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "np": np,
        "json": json,
    }

    try:
        plt.close("all")
        exec(code, safe_globals, safe_locals)
        if "create_chart" not in safe_locals:
            return None, "Функция create_chart не найдена в сгенерированном коде"
        safe_locals["create_chart"](df)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
        buf.seek(0)
        img_bytes = buf.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64, None
    except Exception as e:
        return None, str(e)[:300]
    finally:
        plt.close("all")


def graph_vis(state: dict) -> dict:
    """
    Генерирует код визуализации на основе данных
    """
    logger.info("Запуск ноды graph_vis")
    
    new_state = dict(state)
    
    data = state.get('data')
    if data is None or (isinstance(data, list) and len(data) == 0):
        logger.error("❌ Нет данных для визуализации")
        return {
            **new_state,
            "messages": state.get("messages", []) + [
                AIMessage(content="❌ Нет данных для визуализации")
            ],
            "visualization_code": None,
            "requires_graph_vis": False
        }
    
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
    except Exception as e:
        logger.error(f"❌ Ошибка создания DataFrame: {e}")
        return {
            **new_state,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"❌ Ошибка обработки данных: {str(e)}")
            ],
            "visualization_code": None,
            "requires_graph_vis": False
        }
    
    user_query = state.get("original_query", "Визуализация данных")
    columns = list(df.columns) if hasattr(df, 'columns') else []
    
    prompt = f"""Ты — эксперт по визуализации данных на matplotlib/seaborn.

ТРЕБОВАНИЯ:
1. Сгенерируй ТОЛЬКО Python-функцию с именем create_chart(df)
2. Функция не должна ничего возвращать (void)
3. Используй ТОЛЬКО эти переменные: plt (matplotlib), sns (seaborn), pd (pandas), np (numpy), df (dataframe)
4. В конце ОБЯЗАТЕЛЬНО: plt.savefig('/tmp/chart.png', bbox_inches='tight', dpi=150, facecolor='white')
5. НЕ используй plt.show()
6. НЕ импортируй ничего - все уже импортировано!
7. Если данные пустые → используй plt.text()

ЗАПРЕЩЕНО:
- import (НЕ импортируй!)
- eval, exec, __import__
- os, sys, subprocess, open, requests

ОБЯЗАТЕЛЬНО обработай:
- NaN и None значения
- Пустые данные
- Разные типы графиков (line, bar, scatter, hist, box)

Запрос пользователя: {user_query}
Колонки: {columns}
Строк: {len(df)}

Сгенерируй ТОЛЬКО функцию create_chart(df):"""
    
    max_attempts = 2
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            if attempt == 0:
                logger.info("🤖 Вызов LLM для генерации кода...")
                current_prompt = prompt
            else:
                logger.info(f"🔄 Повторная генерация (попытка {attempt + 1}), ошибка: {last_error}")
                current_prompt = prompt + f"""

ВНИМАНИЕ: Предыдущая попытка сгенерировала код с синтаксической ошибкой:
{last_error}

Пиши максимально простой код. Убедись, что все скобки закрыты. НЕ используй markdown-форматирование."""
            
            response = llm_graph.invoke(current_prompt)
            code = extract_code(response.content.strip())
            
            if "def create_chart" not in code:
                logger.error("❌ LLM не сгенерировал функцию create_chart")
                last_error = "Функция create_chart не найдена"
                continue
            
            syntax_err = _validate_syntax(code)
            if syntax_err:
                logger.warning(f"⚠️ Синтаксическая ошибка в сгенерированном коде: {syntax_err}")
                last_error = syntax_err
                continue
            
            return {
                **new_state,
                "messages": state.get("messages", []) + [
                    AIMessage(content="✅ Код визуализации сгенерирован")
                ],
                "visualization_code": code,
                "data": data, 
                "requires_graph_vis": False 
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка при вызове LLM (попытка {attempt + 1}): {e}")
            last_error = str(e)[:200]
    
    return {
        **new_state,
        "messages": state.get("messages", []) + [
            AIMessage(content=f"❌ Не удалось сгенерировать корректный код визуализации: {last_error}")
        ],
        "visualization_code": None,
        "requires_graph_vis": False
    }


def review_visualization(state: dict) -> dict:
    """
    Human-in-the-loop: даём человеку подтвердить или отклонить код визуализации
    перед выполнением.
    """
    logger.info("Запуск ноды review_visualization (human-in-the-loop)")

    code = state.get("visualization_code")
    data = state.get("data")
    if not code:
        logger.warning("Нет кода визуализации — пропускаем review_visualization")
        return state

    preview = code if len(code) <= 1200 else code[:1200] + "\n# ... truncated ..."

    columns = []
    row_count = 0
    preview_data = []
    chart_preview_base64 = None
    preview_error = None
    try:
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            columns = list(df.columns)
            row_count = len(df)
            preview_data = data[: min(5, len(data))]
            chart_preview_base64, preview_error = _render_chart_base64(code, df)
    except Exception:
        pass

    payload = {
        "type": "visualization_review",
        "code": preview,
        "columns": columns,
        "row_count": row_count,
        "preview_data": preview_data,
        "chart_base64": chart_preview_base64,
        "preview_error": preview_error,
    }

    review_result = interrupt(payload)

    approved = bool(review_result.get("approved", False)) if isinstance(review_result, dict) else False
    updated_code = review_result.get("code") if isinstance(review_result, dict) and review_result.get("code") else code

    if not approved:
        logger.info("Визуализация отклонена человеком")
        return {
            **state,
            "visualization_code": None,
            "messages": state.get("messages", []) + [
                AIMessage(content="❌ Визуализация отклонена пользователем")
            ],
        }

    logger.info("Визуализация одобрена человеком")
    return {
        **state,
        "visualization_code": updated_code,
    }


def safe_exec(state: dict) -> dict:
    """
    Безопасно выполняет сгенерированный код визуализации.
    """
    logger.info("Запуск ноды safe_exec (выполнение кода визуализации)")
    
    new_state = dict(state)
    
    data = state.get('data')
    code = state.get('visualization_code')
    
    if code is None:
        logger.error("❌ Нет кода визуализации")
        return {
            **new_state,
            "messages": state.get("messages", []) + [
                AIMessage(content="❌ Нет кода для выполнения")
            ]
        }
    
    if data is None:
        logger.error("❌ Нет данных")
        return {
            **new_state,
            "messages": state.get("messages", []) + [
                AIMessage(content="❌ Нет данных")
            ]
        }
    
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        logger.info(f"📊 DataFrame создан: {df.shape[0]} строк, {df.shape[1]} колонок")
    except Exception as e:
        logger.error(f"❌ Ошибка при создании DataFrame: {e}")
        return {
            **new_state,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"❌ Ошибка обработки данных: {str(e)}")
            ]
        }
    
    try:
        img_base64, render_error = _render_chart_base64(code, df)
        if not img_base64:
            logger.error(f"❌ Ошибка рендера графика: {render_error}")
            return {
                **new_state,
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"❌ Ошибка выполнения визуализации: {render_error or 'неизвестная ошибка'}")
                ]
            }

        image_markdown = f"![График](data:image/png;base64,{img_base64})"
        
        return {
            **new_state,
            "messages": state.get("messages", []) + [
                AIMessage(
                    content=image_markdown,
                    name="visualization"
                )
            ]
        }
        
    except Exception as e:
        error_type = type(e).__name__
        error_str = str(e)[:300]
        
        if "not defined" in error_str.lower() or "not found" in error_str.lower():
            missing_var = ""
            if "pd" in error_str:
                missing_var = "pandas (pd)"
            elif "np" in error_str:
                missing_var = "numpy (np)"
            elif "plt" in error_str:
                missing_var = "matplotlib (plt)"
            elif "sns" in error_str:
                missing_var = "seaborn (sns)"
            elif "df" in error_str:
                missing_var = "dataframe (df)"
            
            error_msg = f"❌ Ошибка: переменная '{missing_var}' не определена или не доступна. {error_str[:100]}"
        elif "traceback" in error_str.lower() or "syntax" in error_str.lower():
            error_msg = f"❌ Синтаксическая ошибка в коде: {error_str[:100]}"
        else:
            error_msg = f"❌ Ошибка выполнения ({error_type}): {error_str}"
        
        logger.error(f"Ошибка при выполнении кода: {error_msg}")
        logger.debug("Полный стек вызовов:", exc_info=True)
        
        return {
            **new_state,
            "messages": state.get("messages", []) + [
                AIMessage(content=error_msg)
            ]
        }