# Event Snap

A full-stack web application for event photo management with advanced face recognition and AI-powered image organization capabilities.

## 🌟 Features

- **Face Recognition**: Automatically detect and identify faces in photos using DeepFace
- **Intelligent Organization**: AI-powered categorization and organization of event photos
- **Image Management**: Upload, store, and manage photos with thumbnail generation
- **User Authentication**: Secure JWT-based authentication system
- **Real-time Processing**: FAISS-powered vector search for similar faces
- **Responsive UI**: Modern React frontend with Radix UI components
- **REST API**: Comprehensive FastAPI backend with async support

## 🛠️ Tech Stack

### Frontend
- **React** 19.0.0
- **Tailwind CSS** - Utility-first CSS framework
- **Radix UI** - Unstyled, accessible component library
- **React Router** 7.5.1 - Client-side routing
- **Axios** - HTTP client
- **Framer Motion** - Animation library
- **React Hook Form** - Form state management

### Backend
- **FastAPI** - Modern, fast web framework
- **Python 3.x**
- **MongoDB** - Document database
- **Motor** - Async MongoDB driver
- **DeepFace** - Face recognition and detection
- **FAISS** - Facebook AI Similarity Search
- **JWT** - Secure authentication
- **CORS** - Cross-origin resource sharing

### AI & ML
- **Google Generative AI** - LLM integration
- **DeepFace** - Deep learning face recognition
- **FAISS** - Efficient similarity search
- **OpenCV** - Computer vision processing
- **PIL** - Image processing

## 📋 Prerequisites

- Node.js (v16+)
- Python 3.8+
- MongoDB instance
- Git

## 🚀 Getting Started

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/abhishek1165/Event_Snap.git
cd Event_Snap
```

2. Create a `.env` file in the backend directory:
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=event_snap
JWT_SECRET=Enter your -secret -key
```

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python server.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## 📁 Project Structure

```
Event_Snap/
├── backend/
│   ├── server.py           # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
├── frontend/
│   ├── src/               # React components and pages
│   ├── public/            # Static assets
│   ├── package.json       # Node dependencies
│   └── tailwind.config.js # Tailwind configuration
├── storage/               # File storage for photos
├── tests/                 # Test files
└── README.md             # This file
```

## 🔐 Authentication

The application uses JWT-based authentication. Users need to:

1. Register with email and password
2. Login to receive a JWT token
3. Include the token in the `Authorization: Bearer <token>` header for API requests

## 🖼️ Image Processing

- Photos are stored in the `storage/photos` directory
- Thumbnails are automatically generated and stored in `storage/thumbnails`
- FAISS indices are stored in `storage/faiss_indices` for fast similarity search

## 🧪 Testing

Run the test suite:
```bash
python backend_test.py
```

Test results will be saved to `test_result.md`

## 📝 API Endpoints

### Authentication
- `POST /auth/register` - Register a new user
- `POST /auth/login` - Login user
- `POST /auth/refresh` - Refresh JWT token

### Photos
- `GET /photos` - Get all photos
- `POST /photos/upload` - Upload a new photo
- `GET /photos/{photo_id}` - Get photo details
- `DELETE /photos/{photo_id}` - Delete a photo

### Face Recognition
- `POST /recognize/faces` - Detect and recognize faces in an image
- `GET /recognize/similar/{face_id}` - Find similar faces

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Authors

- **Abhishek** - Initial work - [GitHub](https://github.com/abhishek1165)

## 📞 Support

For support, email abhishekkachhap9471@gmail.com or open an issue on the GitHub repository.

## 🚀 Future Enhancements

- [ ] Real-time face detection using WebRTC
- [ ] Advanced search filters
- [ ] Social sharing features
- [ ] Mobile app
- [ ] Event-based organization
- [ ] Automated photo suggestions
- [ ] Integration with cloud storage services

---

**Happy snapping!** 📸
