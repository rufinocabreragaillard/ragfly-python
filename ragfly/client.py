"""RAGfly Python SDK — cliente oficial."""

import json
from typing import Generator, Iterator, Optional, Union
from urllib.parse import urljoin

import httpx

from .models import AskChunk, AskResponse, Chunk, Document, SearchResult

DEFAULT_BASE_URL = "https://api.ragfly.ai"
_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
#: Función de interfaz por defecto para ``ask()`` (define el modelo LLM de la conversación).
DEFAULT_FUNCION = "CHAT-USUARIO"


class RAGflyError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RAGfly:
    """Cliente oficial de RAGfly.

    Uso básico::

        from ragfly import RAGfly

        client = RAGfly(api_key="slm_live_...")
        resp = client.ask("¿Cuáles son las ventas de Q1?")
        print(resp.answer)

    Streaming::

        for chunk in client.ask("¿Cuáles son las ventas de Q1?", stream=True):
            print(chunk.delta, end="", flush=True)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._async_http: Optional[httpx.AsyncClient] = None

    # ── Internos ─────────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RAGflyError(detail, status_code=resp.status_code)

    def _get_or_create_conversation(self, codigo_funcion: str = DEFAULT_FUNCION) -> int:
        """Crea una conversación nueva y devuelve su id."""
        resp = self._http.post(self._url("/interfaz/conversaciones"), json={
            "titulo": "SDK",
            "codigo_funcion": codigo_funcion,
        })
        self._raise_for_status(resp)
        return resp.json()["id_conversacion"]

    # ── API pública ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        min_similitud: float = 0.0,
        codigo_entidad: Optional[str] = None,
        id_espacio: Optional[int] = None,
    ) -> SearchResult:
        """Búsqueda semántica híbrida (vector + léxico + rerank).

        Returns:
            :class:`SearchResult` con lista de documentos y sus chunks relevantes.
        """
        payload = {
            "q": query,
            "limit": limit,
            "min_similitud": min_similitud,
        }
        if codigo_entidad:
            payload["codigo_entidad"] = codigo_entidad
        if id_espacio:
            payload["id_espacio"] = id_espacio

        resp = self._http.post(self._url("/documentos/buscar-semantico"), json=payload)
        self._raise_for_status(resp)
        data = resp.json()

        docs = []
        for d in data.get("resultados", []):
            chunks = [
                Chunk(
                    texto=c.get("texto", ""),
                    similitud=c.get("similitud"),
                    score_rerank=c.get("score_rerank"),
                    # La API expone el nº de página como `nro_pagina` (no `pagina`).
                    pagina=c.get("nro_pagina"),
                    extra={k: v for k, v in c.items() if k not in {"texto", "similitud", "score_rerank", "nro_pagina"}},
                )
                for c in d.get("chunks", [])
            ]
            docs.append(Document(
                codigo=d["codigo_documento"],
                nombre=d["nombre_documento"],
                resumen=d.get("resumen_documento"),
                url=d.get("url"),
                rrf_score=d.get("rrf_score"),
                similitud_max=d.get("similitud_max"),
                chunks=chunks,
            ))

        return SearchResult(
            query=data["q"],
            total_documentos=data["total_documentos"],
            total_chunks=data["total_chunks"],
            duracion_ms=data.get("duracion_ms"),
            documents=docs,
        )

    def ask(
        self,
        question: str,
        *,
        conversation_id: Optional[int] = None,
        codigo_funcion: str = DEFAULT_FUNCION,
        stream: bool = False,
    ) -> Union[AskResponse, Iterator[AskChunk]]:
        """Pregunta al RAG con respuesta completa o streaming.

        Args:
            question: La pregunta en lenguaje natural.
            conversation_id: Reusar conversación existente. Si es None, crea una nueva.
            codigo_funcion: Función de interfaz que define el modelo LLM al crear una
                conversación nueva. Default ``CHAT-USUARIO``. Ignorado si se pasa
                ``conversation_id``.
            stream: Si True, devuelve un iterador de :class:`AskChunk`.

        Returns:
            :class:`AskResponse` (stream=False) o ``Iterator[AskChunk]`` (stream=True).
        """
        conv_id = conversation_id or self._get_or_create_conversation(codigo_funcion)

        if stream:
            return self._ask_stream(question, conv_id)
        return self._ask_sync(question, conv_id)

    def _ask_stream(self, question: str, conv_id: int) -> Iterator[AskChunk]:
        url = self._url(f"/interfaz/conversaciones/{conv_id}/mensajes/stream")
        # En streaming NO aplicamos read-timeout: la generación del LLM puede tardar
        # más que el timeout normal entre tokens, y abortaría el SSE a mitad de
        # respuesta (httpx.ReadTimeout). Mantenemos solo el connect-timeout. Espejo
        # del SDK TS, que cancela su timer al recibir los headers de respuesta.
        with self._http.stream(
            "POST", url, json={"contenido": question},
            timeout=httpx.Timeout(None, connect=10.0),
        ) as resp:
            self._raise_for_status(resp)
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if "error" in payload:
                    raise RAGflyError(payload["error"])
                if payload.get("done"):
                    return
                if "text" in payload:
                    yield AskChunk(delta=payload["text"])

    def _ask_sync(self, question: str, conv_id: int) -> AskResponse:
        buffer = []
        msg_id = None
        for chunk in self._ask_stream(question, conv_id):
            buffer.append(chunk.delta)
        # El done payload lleva id_mensaje_assistant pero lo emitimos antes de salir
        # del generator — capturamos el último evento done fuera del yield.
        return AskResponse(
            answer="".join(buffer),
            conversation_id=conv_id,
            message_id=msg_id,
        )

    def list_documents(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        estado: Optional[str] = None,
    ) -> dict:
        """Lista documentos del corpus con paginación."""
        # El backend (GET /documentos/paginado) espera page/limit/codigo_estado_doc.
        # FastAPI ignora params desconocidos, así que pagina/limite/estado se
        # traducían en "sin filtro, 50 por defecto".
        params: dict = {"page": page, "limit": page_size}
        if estado:
            params["codigo_estado_doc"] = estado
        resp = self._http.get(self._url("/documentos/paginado"), params=params)
        self._raise_for_status(resp)
        return resp.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
