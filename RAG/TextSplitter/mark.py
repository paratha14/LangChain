markdown="""# 🚀 Project Name

A brief description of your project.  
For example: A powerful and minimalistic web application that lets users upload and analyze PDF documents in real-time.

---

## 📸 Demo

---

## 🔧 Features

- ✅ Upload and preview PDF documents
- ⚡ Real-time analysis with instant results
- 🌐 Built with [Your Tech Stack Here]
- 📦 Lightweight and responsive UI

---

## 🛠️ Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Node.js / Python / Flask / Express (modify as needed)
- Deployment: Render / Vercel / Heroku / GitHub Pages

---

## 🚀 Getting Started

Clone the repo:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size= 30,
    chunk_overlap=0,
)

chunks= splitter.split_text(markdown)
print(chunks)