'''
Определение состояния агента для LangGraph
'''

from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    '''
    Поля:
    - messages: История сообщений (накапливаются через operator.add)
    - query_result: Результат SQL-запроса (list[dict])
    - last_sql: Последний выполненный SQL-запрос
    - from_cache: Был ли результат получен из кэша
    - data: Данные для визуализации (list[dict])
    - original_query: Исходный запрос пользователя
    - visualization_code: Сгенерированный код визуализации
    - requires_graph_vis: Нужна ли визуализация (флаг для переходов)
    - sql_error_count: общее количество ошибок SQL
    - consecutive_same_errors: Сколько раз подряд одна и та же ошибка
    - last_error_message: текст последней ошибки
    - critic_ran_last: Запускался ли критик
    - critic_attempts: Количество попыток критика
    - cache_reject_query: Запрос, для которого пользователь отклонил кэш (нужен для замены)
    '''

    messages: Annotated[Sequence[BaseMessage], operator.add]
    query_result: Optional[list]
    last_sql: Optional[str]
    from_cache: Optional[bool]
    data: Optional[list]
    original_query: Optional[str]
    visualization_code: Optional[str]
    requires_graph_vis: Optional[bool]
    sql_error_count: int
    consecutive_same_errors: int
    last_error_message: Optional[str]
    critic_ran_last: bool
    critic_attempts: int
    cache_reject_query: Optional[str]
