from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .service import CorpusService
from .ui import INDEX_HTML


class AskRequest(BaseModel):
    question: str
    topic: str = 'graph neural network'
    top_k: int = 5
    per_source: int = 8
    sources: Optional[List[str]] = None


def create_app(service: Optional[CorpusService] = None) -> FastAPI:
    svc = service or CorpusService()
    app = FastAPI(
        title='LitScope',
        version='0.1.0',
        description='Multi-source literature retrieval with citation-grounded answers',
    )

    @app.get('/', response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get('/api/health')
    def health() -> dict:
        return {
            'status': 'ok',
            'embedder': svc.embedder.name,
            'cached_corpora': len(svc.cache),
            'llm': 'ready' if svc.llm_ready else 'not configured',
        }

    @app.get('/api/search')
    def search(q: str = Query(..., min_length=2), limit: int = 10,
               sources: str = 'arxiv,crossref,pubmed') -> dict:
        chosen = [name.strip() for name in sources.split(',') if name.strip()]
        try:
            return svc.search(q, limit=limit, sources=chosen)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post('/api/ask')
    def ask(request: AskRequest) -> dict:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail='question must not be empty')
        try:
            return svc.ask(request.topic, request.question, top_k=request.top_k,
                           per_source=request.per_source, sources=request.sources)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)[:300]) from exc

    return app


app = create_app()
