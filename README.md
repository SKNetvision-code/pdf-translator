# PDF Translator - English to Hindi

A web application that translates English PDFs to Hindi using Google Translate.

## Features

- Upload PDF files (up to 1000 pages)
- Translates text content to Hindi
- Preserves images and graphs in the PDF
- Downloads translation as text file

## How to Use

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

### Deploy to Render (Free)

1. Push this project to GitHub
2. Sign up at [render.com](https://render.com)
3. Create a new **Web Service**
4. Connect your GitHub repo
5. Use the `render.yaml` settings (already configured)
6. Click **Deploy**

Your app will be live at `https://yourapp.onrender.com`

### How It Works

1. Upload a PDF file via drag & drop or file browser
2. The app extracts text using pdfplumber
3. Text is translated to Hindi in chunks using deep_translator (Google Translate)
4. Images are preserved using PyMuPDF
5. Download the translated text file when complete

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/upload` | POST | Upload PDF file |
| `/status/{job_id}` | GET | Check translation status |
| `/download/{job_id}` | GET | Download translation |
| `/images/{job_id}` | GET | Download preserved images |

## Files

```
pdf-translator/
├── app.py              # FastAPI application
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── static/
│   └── style.css       # Frontend styles
├── templates/
│   └── index.html      # Frontend HTML
├── uploads/            # Uploaded PDFs (gitignored)
├── translations/       # Translation output (gitignored)
└── images/             # Extracted images (gitignored)
```