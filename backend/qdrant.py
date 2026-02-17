import asyncio
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langgraph.graph import END
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from config import QDRANT_URL, QDRANT_API
from database import get_cache_namespace

logger = logging.getLogger(__name__)

vectorstore = None
embeddings = None
qdrant_client = None
_vectorstores: dict[str, QdrantVectorStore] = {}
_reconnect_task = None


def _init_embeddings():
    """Инициализирует embeddings модель (тяжёлая операция, делается один раз)."""
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/LaBSE",
            encode_kwargs={"normalize_embeddings": True}
        )
    return embeddings


def _connect_qdrant():
    """Пытается подключиться к Qdrant. Возвращает vectorstore или None."""
    global vectorstore, qdrant_client
    try:
        emb = _init_embeddings()
        if emb is None:
            return None

        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API, timeout=10)
        vectorstore = get_active_vectorstore()
        logger.info("✅ Qdrant готов (динамические коллекции по БД)")
        return vectorstore

    except Exception as e:
        logger.error(f"❌ Ошибка подключения к Qdrant: {e}")
        vectorstore = None
        qdrant_client = None
        _vectorstores.clear()
        return None


def init_vectorstore_async():
    """Инициализация Qdrant + HuggingFace embeddings. Возвращает (embeddings, vectorstore) или (None, None)."""
    global vectorstore, embeddings
    _connect_qdrant()
    return embeddings, get_active_vectorstore()


def _ensure_collection(collection_name: str):
    if qdrant_client is None:
        return
    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )


def get_active_vectorstore() -> QdrantVectorStore | None:
    """Возвращает vectorstore текущей активной БД (отдельная коллекция на БД)."""
    global vectorstore
    if qdrant_client is None or embeddings is None:
        return None

    try:
        collection_name = get_cache_namespace()
        if collection_name in _vectorstores:
            vectorstore = _vectorstores[collection_name]
            return vectorstore

        _ensure_collection(collection_name)
        store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        _vectorstores[collection_name] = store
        vectorstore = store
        logger.info(f"🗂 Активная коллекция кэша: {collection_name}")
        return store
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации коллекции кэша: {e}")
        return None


def get_active_collection_name() -> str:
    """Имя активной коллекции кэша для текущей БД."""
    return get_cache_namespace()


async def _background_reconnect(interval: int = 30, max_attempts: int = 0):
    """Фоновая задача переподключения к Qdrant.
    
    interval: секунды между попытками
    max_attempts: 0 = бесконечно
    """
    attempt = 0
    while True:
        if qdrant_client is not None and embeddings is not None:
            await asyncio.sleep(interval)
            continue

        attempt += 1
        if max_attempts and attempt > max_attempts:
            logger.warning(f"⚠️ Достигнут лимит попыток переподключения к Qdrant ({max_attempts})")
            return

        logger.info(f"🔄 Попытка переподключения к Qdrant #{attempt}...")
        try:
            result = await asyncio.get_running_loop().run_in_executor(None, _connect_qdrant)
            if result is not None:
                logger.info("✅ Qdrant переподключён! Кэширование включено.")
                return
        except Exception as e:
            logger.error(f"❌ Попытка #{attempt} не удалась: {e}")

        await asyncio.sleep(interval)


def start_reconnect_task(interval: int = 30):
    """Запускает фоновую задачу переподключения, если Qdrant недоступен."""
    global _reconnect_task
    if qdrant_client is not None and embeddings is not None:
        return
    if _reconnect_task and not _reconnect_task.done():
        return
    logger.info("🔄 Запуск фоновой задачи переподключения к Qdrant...")
    _reconnect_task = asyncio.create_task(_background_reconnect(interval=interval))


def _fresh_state_update() -> dict:
    """Базовый dict-обновление, сбрасывающий per-request поля.

    MemorySaver хранит состояние между запросами в рамках одного thread_id.
    Без явного сброса поля вроде from_cache / query_result / data
    «протекают» из предыдущего запроса и ломают маршрутизацию.
    """
    return {
        "from_cache": False,
        "query_result": None,
        "data": None,
        "visualization_code": None,
        "requires_graph_vis": False,
        "critic_attempts": 0,
        "critic_ran_last": False,
        "sql_error_count": 0,
        "consecutive_same_errors": 0,
        "last_error_message": None,
        "last_sql": None,
        "cache_reject_query": None,
    }


def checked_cache(state: AgentState):
    """Проверяет кэш Qdrant. При hit — предлагает пользователю выбор (interrupt).

    Возвращает ЧАСТИЧНОЕ обновление состояния (не полный state),
    чтобы operator.add-редьюсер для messages не дублировал историю.
    """
    global embeddings

    updates = _fresh_state_update()

    store = get_active_vectorstore()
    if store is None or embeddings is None:
        logger.warning('Qdrant или embeddings не готовы, пропуск кэша')
        return updates

    messages = state.get("messages", [])

    query_msg = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            query_msg = m
            break

    query = query_msg.content.strip() if query_msg else ""
    if not query and messages:
        query = messages[-1].content.strip()

    if query:
        updates["original_query"] = query

    if not query:
        return updates

    try:
        results = store.similarity_search_with_score(query=query, k=1)

        if results:
            doc, score = results[0]
            if score >= 0.95:
                logger.info(f"✅ Кэш hit! Score: {score:.3f}")
                cached_response = doc.metadata.get('response', doc.page_content)
                cached_data = doc.metadata.get('data', [])

                payload = {
                    "type": "cache_review",
                    "query": query,
                    "cached_response": cached_response,
                    "cached_data": cached_data,
                    "score": float(score),
                }

                decision = interrupt(payload)

                use_cache = isinstance(decision, dict) and bool(decision.get("use_cache", False))

                if use_cache:
                    logger.info("⚡ Пользователь одобрил использование кэша")
                    updates["messages"] = [SystemMessage(content=cached_response)]
                    updates["from_cache"] = True
                    updates["query_result"] = cached_data or []
                    return updates

                logger.info("♻ Пользователь предпочёл сгенерировать ответ заново")
                updates["cache_reject_query"] = query
                return updates
            else:
                logger.info(f"Близкий кандидат найден, но score {score:.3f} < 0.95 → miss")
        else:
            logger.info("❌ Кэш miss (нет кандидатов)")

        return updates

    except GraphInterrupt:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка при поиске в Qdrant: {type(e).__name__} → {str(e)}")
        return updates


def delete_cache_entry(query: str):
    """Находит и удаляет ближайшую запись кэша для данного запроса."""
    store = get_active_vectorstore()
    if not store or not embeddings:
        return

    try:
        query_vector = embeddings.embed_query(query)
        client = store.client
        collection_name = store.collection_name
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=1,
        )

        if search_results.points:
            point = search_results.points[0]
            if point.score >= 0.90:
                client.delete(
                    collection_name=collection_name,
                    points_selector=[point.id],
                )
                logger.info(f"🗑 Удалена старая запись кэша (ID: {point.id}, score: {point.score:.3f})")
            else:
                logger.info(f"Ближайшая запись score {point.score:.3f} < 0.90 — не удаляем")
        else:
            logger.info("Нет записей для удаления")
    except Exception as e:
        logger.error(f"❌ Ошибка при удалении записи кэша: {e}")


def cache_should_continue(state: AgentState) -> str:
    """Если ответ из кэша — END, иначе — assistant."""
    if state.get("from_cache"):
        return END
    return "assistant"
