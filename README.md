# 🎙️ Transcriber Meeting - AI-Powered Voice Conversation Platform

A production-ready full-stack application for recording, transcribing, diarizing, and summarizing multi-speaker conversations with intelligent voice profile management and AI-powered insights.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![React](https://img.shields.io/badge/react-18+-61dafb.svg)
![TypeScript](https://img.shields.io/badge/typescript-5+-3178c6.svg)

---

## ✨ Features

### 🔐 Authentication & User Management
- JWT-based secure authentication
- User registration and login
- Voice profile onboarding (3-sample setup)
- Multi-user voice profile management

### 🎤 Recording & Processing
- **Live Recording**: Browser-based real-time audio capture
- **File Upload**: Support for pre-recorded audio files
- **Real-time Transcription**: Powered by faster-whisper
- **Speaker Diarization**: 
  - Advanced: pyannote.audio (optional, requires HuggingFace token)
  - Fallback: Energy-based diarization
- **Speaker Identification**: WeSpeaker CAM++ embeddings for voice matching
- **Background Processing**: Async pipeline for efficient processing

### 🤖 AI-Powered Insights
- **Intelligent Summaries**: Groq LLM-powered conversation summaries
- **Key Points Extraction**: Automatic identification of important topics
- **Action Items**: AI-detected tasks and follow-ups
- **Interactive Q&A**: Chat with your transcripts using AI
- **Confidence Scoring**: Word-level confidence visualization

### 📊 Management & History
- Recording history with search and filters
- Voice profile management (add/rename/delete speakers)
- Configurable AI and processing thresholds
- Session-based organization
- Export capabilities

---

## 🛠️ Tech Stack

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS + shadcn/ui components
- **State Management**: Zustand
- **HTTP Client**: Axios
- **Routing**: React Router

### Backend
- **Framework**: FastAPI (Python)
- **Database**: MongoDB with Motor (async driver)
- **Authentication**: JWT tokens
- **AI Models**:
  - faster-whisper (transcription)
  - WeSpeaker CAM++ / onnxruntime (voice embeddings)
  - pyannote.audio (optional diarization)
  - Groq LLM (summaries & Q&A)
- **Audio Processing**: librosa, pydub, webrtcvad
- **Storage**: Local filesystem

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB instance
- Groq API key ([Get one here](https://console.groq.com))
- (Optional) HuggingFace token for pyannote.audio

### 1️⃣ Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install pyannote for advanced diarization
pip install pyannote.audio

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required: MONGODB_URI, JWT_SECRET, GROQ_API_KEY
```

**Start the backend server:**
```bash
uvicorn main:app --reload --port 8000
```

Backend will be available at: `http://localhost:8000`  
API Documentation: `http://localhost:8000/docs`

### 2️⃣ Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Copy environment template (if needed)
cp .env.example .env

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:5173`

---

## ⚙️ Configuration

### Backend Environment Variables (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MONGODB_URI` | ✅ Yes | - | MongoDB connection string |
| `JWT_SECRET` | ✅ Yes | - | Secret key for JWT token signing |
| `GROQ_API_KEY` | ✅ Yes | - | Groq API key for AI features |
| `GROQ_MODEL` | No | `llama3-8b-8192` | Groq model to use |
| `HF_TOKEN` | No | - | HuggingFace token (enables pyannote) |
| `WHISPER_MODEL_SIZE` | No | `medium` | Whisper model size (tiny/base/small/medium/large) |
| `WHISPER_DEVICE` | No | `auto` | Device for inference (auto/cuda/cpu) |
| `SPEAKER_SIMILARITY_THRESHOLD` | No | `0.75` | Threshold for speaker matching (0-1) |

### Frontend Environment Variables (`frontend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_API_URL` | No | `http://localhost:8000` | Backend API URL |

---

## 📁 Project Structure

```
transcriber-meeting/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── database.py             # MongoDB connection
│   ├── requirements.txt        # Python dependencies
│   ├── models/                 # Database models
│   │   ├── user.py
│   │   ├── recording.py
│   │   └── settings.py
│   ├── routers/                # API endpoints
│   │   ├── auth.py
│   │   ├── audio.py
│   │   ├── voice.py
│   │   ├── history.py
│   │   ├── chat.py
│   │   └── settings_router.py
│   ├── services/               # Business logic
│   │   ├── transcription.py
│   │   ├── diarization.py
│   │   ├── identification.py
│   │   ├── embedding.py
│   │   ├── llm.py
│   │   └── record.py
│   ├── tasks/                  # Background processing
│   │   └── pipeline.py
│   └── utils/                  # Utilities
│       ├── audio_utils.py
│       └── storage.py
│
└── frontend/
    ├── src/
    │   ├── api/                # API client
    │   │   └── client.ts
    │   ├── components/         # React components
    │   │   ├── recording/
    │   │   ├── session/
    │   │   ├── dashboard/
    │   │   ├── shared/
    │   │   └── ui/
    │   ├── hooks/              # Custom React hooks
    │   │   ├── useAudioRecorder.ts
    │   │   └── useJobPoller.ts
    │   ├── pages/              # Page components
    │   │   ├── Login.tsx
    │   │   ├── Signup.tsx
    │   │   ├── Setup.tsx
    │   │   ├── Dashboard.tsx
    │   │   ├── Record.tsx
    │   │   ├── History.tsx
    │   │   └── Settings.tsx
    │   ├── store/              # State management
    │   │   ├── auth.ts
    │   │   └── ui.ts
    │   └── App.tsx
    ├── package.json
    └── vite.config.ts
```

---

## 🎯 Usage

### 1. Create an Account
- Navigate to the signup page
- Create your account with email and password

### 2. Voice Profile Setup
- Record 3 voice samples (5-10 seconds each)
- System creates your unique voice profile
- Add additional speakers as needed

### 3. Record or Upload
- **Live Recording**: Click record button and start speaking
- **Upload**: Drag and drop audio files (WAV, MP3, M4A)

### 4. Processing
- Automatic transcription with speaker labels
- AI-generated summary, key points, and action items
- Interactive transcript with confidence scores

### 5. Review & Chat
- View color-coded transcripts by speaker
- Ask questions about the conversation
- Export or share results

---

## 🔧 Development

### Backend Development

```bash
# Run with auto-reload
uvicorn main:app --reload --port 8000

# Run tests (if available)
pytest

# Format code
black .
```

### Frontend Development

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

---

## 📊 API Documentation

Once the backend is running, visit the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient transcription
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [WeSpeaker](https://github.com/modelscope/3D-Speaker) for CAM++ voice embeddings
- [Groq](https://groq.com) for fast LLM inference
- [shadcn/ui](https://ui.shadcn.com/) for beautiful UI components

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for better conversations**
