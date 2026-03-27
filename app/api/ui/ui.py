from fastapi import APIRouter
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

router = APIRouter()

# app/api/ui/
UI_ROOT = Path(__file__).resolve().parent


@router.get("/ui", response_class=HTMLResponse)
def chat_ui():
    return FileResponse(UI_ROOT / "index.html")


@router.get("/ui/styles.css")
def ui_css():
    return FileResponse(UI_ROOT / "styles.css")


@router.get("/ui/chat.js")
def ui_js():
    return FileResponse(UI_ROOT / "chat.js")
