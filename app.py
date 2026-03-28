import os
import uuid
import time
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from deep_translator import GoogleTranslator
import pdfplumber
import fitz  # PyMuPDF for image extraction

app = FastAPI(title="PDF Translator", description="Translate English PDFs to Hindi")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TRANSLATION_DIR = BASE_DIR / "translations"
IMAGES_DIR = BASE_DIR / "images"

UPLOAD_DIR.mkdir(exist_ok=True)
TRANSLATION_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

translator = GoogleTranslator(source="en", target="hi")


class UploadResponse(BaseModel):
    job_id: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    pages_completed: int
    total_pages: int
    error: Optional[str] = None


jobs = {}


def extract_images_from_pdf(pdf_path: str, job_id: str) -> dict:
    """Extract images from PDF using PyMuPDF and save them."""
    images_info = {}
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha < 4:
                    img_data = pix.tobytes("png")
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix.tobytes("png")
                img_filename = f"{job_id}_page{page_num + 1}_img{img_index + 1}.png"
                img_path = IMAGES_DIR / img_filename
                with open(img_path, "wb") as f:
                    f.write(img_data)
                if page_num + 1 not in images_info:
                    images_info[page_num + 1] = []
                images_info[page_num + 1].append({
                    "filename": img_filename,
                    "index": img_index + 1
                })
        doc.close()
    except Exception as e:
        print(f"Image extraction error: {e}")
    return images_info


def translate_pdf_background(job_id: str, file_path: str):
    """Extract text and images from PDF and translate to Hindi (runs in background)."""
    try:
        jobs[job_id]["status"] = "processing"

        # First pass: get total pages and extract images
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

        # Extract images using PyMuPDF
        images_info = extract_images_from_pdf(file_path, job_id)

        jobs[job_id]["total_pages"] = total_pages

        translated_text = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                page_has_images = page_num in images_info and len(images_info[page_num]) > 0

                if text.strip() or page_has_images:
                    # Handle page with text
                    if text.strip():
                        max_chunk_size = 4500
                        chunks = []
                        paragraphs = text.split("\n\n")

                        current_chunk = ""
                        for para in paragraphs:
                            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                                current_chunk += para + "\n\n"
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                while len(para) > max_chunk_size:
                                    chunks.append(para[:max_chunk_size])
                                    para = para[max_chunk_size:]
                                current_chunk = para + "\n\n"

                        if current_chunk:
                            chunks.append(current_chunk.strip())

                        page_translation = ""
                        for chunk in chunks:
                            try:
                                result = translator.translate(chunk)
                                page_translation += result + "\n\n"
                                time.sleep(0.3)
                            except Exception as e:
                                print(f"Translation error on page {page_num}: {e}")
                                page_translation += f"[Translation Error: {str(e)}]\n\n"
                    else:
                        page_translation = ""

                    # Add image placeholder if page has images
                    if page_has_images:
                        if page_translation.strip():
                            translated_text.append(f"--- Page {page_num} ---\n\n{page_translation}")
                        translated_text.append(f"[Image(s) on this page preserved: {', '.join([img['filename'] for img in images_info[page_num]])}]")
                    else:
                        if page_translation.strip():
                            translated_text.append(f"--- Page {page_num} ---\n\n{page_translation}")
                        else:
                            translated_text.append(f"--- Page {page_num} ---\n\n[Visual content - no extractable text]")
                else:
                    translated_text.append(f"--- Page {page_num} ---\n\n[No text found on this page]")

                progress = (page_num / total_pages) * 100
                jobs[job_id].update({
                    "progress": round(progress, 2),
                    "pages_completed": page_num
                })

        output_path = TRANSLATION_DIR / f"{job_id}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("ENGLISH TO HINDI TRANSLATION\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(translated_text))

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100.0

    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only PDF files are supported"}
        )

    job_id = str(uuid.uuid4())

    file_path = UPLOAD_DIR / f"{job_id}.pdf"
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            if not content:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Empty file uploaded"}
                )
            f.write(content)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to save file: {str(e)}"}
        )

    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "pages_completed": 0,
        "total_pages": 0,
        "file_path": str(file_path),
        "error": None
    }

    background_tasks.add_task(translate_pdf_background, job_id, str(file_path))

    return JSONResponse(content={"job_id": job_id, "message": "PDF uploaded successfully, translation started"})


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"detail": "Job not found"}
        )

    job = jobs[job_id]
    return JSONResponse(content={
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "pages_completed": job["pages_completed"],
        "total_pages": job["total_pages"],
        "error": job["error"]
    })


@app.get("/download/{job_id}")
async def download_translation(job_id: str):
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"detail": "Job not found"}
        )

    job = jobs[job_id]
    if job["status"] != "completed":
        return JSONResponse(
            status_code=400,
            content={"detail": "Translation not yet complete"}
        )

    translation_path = TRANSLATION_DIR / f"{job_id}.txt"
    if not translation_path.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "Translation file not found"}
        )

    return FileResponse(
        path=translation_path,
        filename=f"translation_{job_id}.txt",
        media_type="text/plain; charset=utf-8"
    )


@app.get("/images/{job_id}")
async def download_images(job_id: str):
    """Download all preserved images from a PDF translation as a zip file."""
    import zipfile
    from io import BytesIO

    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"detail": "Job not found"}
        )

    job = jobs[job_id]
    if job["status"] != "completed":
        return JSONResponse(
            status_code=400,
            content={"detail": "Translation not yet complete"}
        )

    # Find all images for this job
    image_files = list(IMAGES_DIR.glob(f"{job_id}_*.png"))

    if not image_files:
        return JSONResponse(
            status_code=404,
            content={"detail": "No images found in this PDF"}
        )

    # Create zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_path in sorted(image_files):
            zipf.write(img_path, img_path.name)

    zip_buffer.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=images_{job_id}.zip"}
    )
