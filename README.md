# Event Snap 📸

> **AI-powered event photo sharing** — Attendees scan a QR code, take one selfie, and instantly find every photo they appear in across thousands of event shots.

---

## ✨ Features

- **⚡ Instant Face Matching** — Take a selfie and FAISS vector search finds every photo of you in seconds
- **🎯 Organizer Dashboard** — Create events, upload photos, track processing stats in real time
- **🖼️ QR Table Card Generator** — Print beautiful QR cards for wedding/event tables (4 templates)
- **📺 Live Slideshow Mode** — Full-screen auto-rotating presentation for venue TVs and projectors
- **🔐 JWT Authentication** — Secure login/register for organizers and attendees
- **☁️ MongoDB GridFS Storage** — All photos, thumbnails, and FAISS indices stored in MongoDB (no local disk required)
- **🎨 Modern Dark Cinema UI** — Premium design with Framer Motion animations and GSAP effects

---

## 📸 Screenshots

### Landing Page
![Landing Page](frontend/public/Screenshot%202026-08-09%20123631.png)

### Organizer Dashboard
![Organizer Dashboard](frontend/public/Screenshot%202026-08-09%20123901.png)

### Event Management & Photo Upload
![Event Management](frontend/public/Screenshot%202026-08-09%20123949.png)

### Attendee Selfie Search
![Selfie Search](frontend/public/Screenshot%202026-08-09%20124028.png)

### Face Recognition Results
![Match Results](frontend/public/Screenshot%202026-08-09%20140248.png)

---

## 🛠️ Tech Stack

### Frontend
- **React 19** + React Router v7
- **Tailwind CSS v3.4** — custom design tokens (`#08090d` ink bg, `#6d5ef5` iris accent, `#34d399` match green)
- **Framer Motion** + **GSAP** — animations and micro-interactions
- **Radix UI** — accessible component primitives
- **qrcode.react** — QR code generation for table cards
- **Sonner** — toast notifications
- **Axios** — API client

### Backend
- **FastAPI** (Python 3.11) — async REST API
- **Motor** — async MongoDB driver
- **MongoDB Atlas** — cloud database + GridFS for file storage
- **DeepFace** (Facenet512) — face embedding extraction
- **FAISS** — sub-second similarity vector search
- **bcrypt + PyJWT** — password hashing and authentication
- **certifi** — SSL/TLS for MongoDB Atlas connections

---

## 📋 Prerequisites

- Node.js v18+
- Python 3.11+
- MongoDB (local or Atlas cloud)
- Git

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/abhishek1165/Event_Snap.git
cd Event_Snap
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Create `backend/.env`:
```env
MONGO_URL=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/event_snap?retryWrites=true&w=majority&appName=Cluster0
DB_NAME=event_snap
JWT_SECRET=your-secret-key-here
CORS_ORIGINS=http://localhost:3000
```

> **Using local MongoDB instead?** Set `MONGO_URL=mongodb://localhost:27017`

Start the backend:
```bash
# Windows (recommended)
.\run_server.bat

# Or directly with uvicorn
uvicorn server:app --host 127.0.0.1 --port 8081

# Or with Python
python server.py
```

Backend runs at: `http://localhost:8081`
Swagger docs at: `http://localhost:8081/docs`

### 3. Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env`:
```env
REACT_APP_BACKEND_URL=http://localhost:8081
```

Start the dev server:
```bash
npm start
```

Frontend runs at: `http://localhost:3000`

### 🐳 Docker & Docker Compose Setup (One-Click Launch)

Run both backend and frontend using Docker Compose:

```bash
# Build & start all containers (backend, frontend, local mongo)
docker-compose up --build -d

# Stop containers
docker-compose down
```

- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8081`
- **Swagger Docs**: `http://localhost:8081/docs`

---

## 📁 Project Structure

```
Event_Snap/
├── backend/
│   ├── server.py              # FastAPI app — all API routes
│   ├── run_server.bat         # Windows startup script (port 8081)
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables (not committed)
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── brand/atoms.jsx                     # Core UI primitives
    │   │   └── organizer/QrCardGeneratorModal.jsx  # QR Table Card Generator
    │   ├── design/tokens.js                         # Color & motion tokens
    │   ├── pages/
    │   │   ├── Landing.js           # Public landing page
    │   │   ├── Auth.js              # Login / Register
    │   │   ├── OrganizerDashboard.js # Event management dashboard
    │   │   ├── EventDetails.js      # Event detail & photo upload
    │   │   ├── AttendeeEntry.js     # Attendee access code entry
    │   │   ├── SelfieCapture.js     # Selfie capture & face search
    │   │   ├── PhotoGallery.js      # Matched photo results
    │   │   └── LiveSlideshow.js     # Venue TV slideshow mode
    │   └── utils/api.js             # Axios instance & backend URL config
    ├── .env                         # Frontend env vars (not committed)
    ├── craco.config.js              # Webpack / devServer config
    ├── tailwind.config.js           # Tailwind v3 theme
    └── package.json
```

---

## 🔐 Authentication

JWT-based authentication with two roles:

| Role | Access |
|------|--------|
| **Organizer** | Create events, upload photos, view dashboard, generate QR cards |
| **Attendee** | Enter event code, submit selfie, view matched photos |

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Register new user |
| `POST` | `/api/auth/login` | Login, receive JWT |
| `GET` | `/api/events` | List organizer's events |
| `POST` | `/api/events` | Create new event |
| `POST` | `/api/events/{id}/photos` | Upload photos to event |
| `GET` | `/api/events/code/{code}` | Attendee lookup by 8-char code |
| `POST` | `/api/search/selfie` | Submit selfie for face-match search |
| `GET` | `/api/events/{id}/photos` | List event photos |
| `DELETE` | `/api/events/{id}/photos/{photo_id}` | Delete photo + rebuild FAISS index |

---

## ⚠️ Windows Port Note

Windows Hyper-V reserves port range `7910–8009`. The backend runs on **port 8081** to bypass this restriction. If you need a different port:

```bash
uvicorn server:app --host 127.0.0.1 --port <YOUR_PORT>
```

And update `frontend/.env`:
```env
REACT_APP_BACKEND_URL=http://localhost:<YOUR_PORT>
```

---

## 👨‍💻 Author

**Abhishek** — [GitHub](https://github.com/abhishek1165) — abhishekkachha12345@gmail.com

---

**Happy Snapping!** 📸
