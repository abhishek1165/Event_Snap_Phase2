from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
import aiofiles
from deepface import DeepFace
import numpy as np
import faiss
import pickle
import cv2
from PIL import Image
import io
import secrets
import asyncio
import hashlib
from bson import ObjectId


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT_DIR / 'event_snap.log')
    ]
)
logger = logging.getLogger(__name__)

import certifi

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
mongo_kwargs = {}
if "mongodb+srv://" in mongo_url:
    mongo_kwargs["tlsCAFile"] = certifi.where()
client = AsyncIOMotorClient(mongo_url, **mongo_kwargs)
db = client[os.environ['DB_NAME']]

# ==================== GRIDFS SETUP (Cloud Storage Alternative) ====================
# GridFS buckets for storing files in MongoDB instead of local filesystem
# This makes the app production-ready for Heroku/Railway/Vercel
photos_fs = AsyncIOMotorGridFSBucket(db, bucket_name="photos")
thumbnails_fs = AsyncIOMotorGridFSBucket(db, bucket_name="thumbnails")
faiss_fs = AsyncIOMotorGridFSBucket(db, bucket_name="faiss_indices")

# Temporary directory for DeepFace processing (DeepFace requires file paths)
TEMP_DIR = Path("/tmp/event_snap") if os.path.exists("/tmp") else Path("./temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# JWT settings
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'

# Face model: same for Organizer (photo indexing) and Attendee (selfie search).
FACE_MODEL = os.environ.get('FACE_MODEL', 'Facenet')

# Detector for group photos: retinaface finds all faces; opencv often finds only 1
FACE_DETECTOR = os.environ.get('FACE_DETECTOR', 'retinaface')  # retinaface, mtcnn, opencv

# Face matching thresholds (lower = more photos returned for same person in different shots)
MINIMUM_MATCH_THRESHOLD = float(os.environ.get('MINIMUM_MATCH_THRESHOLD', '0.35'))
HIGH_CONFIDENCE_THRESHOLD = float(os.environ.get('HIGH_CONFIDENCE_THRESHOLD', '0.7'))

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# In-memory FAISS indices (event_id -> index)
faiss_indices = {}
face_mappings = {}  # event_id -> {face_id -> [photo_ids]}

# ==================== OPTIMIZATION 1: Selfie Embedding Cache ====================
# Cache selfie embeddings by SHA-256 hash of image bytes
# Key: hash of selfie bytes, Value: normalized embedding (np.ndarray)
selfie_embedding_cache = {}

def hash_image_bytes(image_bytes: bytes) -> str:
    """Generate SHA-256 hash of image bytes for caching"""
    return hashlib.sha256(image_bytes).hexdigest()

# ==================== GRIDFS HELPER FUNCTIONS ====================

async def upload_to_gridfs(fs_bucket, file_content: bytes, filename: str, metadata: dict = None) -> str:
    """Upload file to GridFS and return the file ID"""
    file_id = await fs_bucket.upload_from_stream(
        filename,
        file_content,
        metadata=metadata or {}
    )
    return str(file_id)

async def download_from_gridfs(fs_bucket, file_id: str) -> bytes:
    """Download file from GridFS by ID"""
    try:
        grid_out = await fs_bucket.open_download_stream(ObjectId(file_id))
        contents = await grid_out.read()
        return contents
    except Exception as e:
        logging.error(f"Error downloading from GridFS: {e}")
        raise HTTPException(status_code=404, detail="File not found")

async def delete_from_gridfs(fs_bucket, file_id: str):
    """Delete file from GridFS"""
    try:
        await fs_bucket.delete(ObjectId(file_id))
    except Exception as e:
        logging.warning(f"Error deleting from GridFS: {e}")

# ==================== MODELS ====================

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "attendee"  # organizer or attendee

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    name: str
    role: str
    created_at: datetime

class EventCreate(BaseModel):
    title: str
    description: Optional[str] = None
    date: Optional[str] = None

class Event(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    organizer_id: str
    title: str
    description: Optional[str] = None
    event_code: str
    date: Optional[str] = None
    status: str = "active"  # active, processing, completed
    total_photos: int = 0
    processed_photos: int = 0
    faces_detected: int = 0
    created_at: datetime

class Photo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    event_id: str
    gridfs_id: str  # GridFS file ID instead of file_path
    thumbnail_gridfs_id: str  # GridFS thumbnail ID
    upload_status: str = "uploaded"  # uploaded, processing, completed, failed
    processed: bool = False
    faces_detected: int = 0
    created_at: datetime

class FaceEmbedding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    photo_id: str
    event_id: str
    face_id: str
    bbox: List[float]
    created_at: datetime

class SearchResult(BaseModel):
    photo_id: str
    event_id: str
    similarity_score: float
    thumbnail_url: str
    photo_url: str

class ProcessingStatus(BaseModel):
    event_id: str
    total_photos: int
    processed_photos: int
    faces_detected: int
    status: str

# ==================== AUTH UTILS ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await db.users.find_one({"id": user_id}, {"_id": 0})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        if isinstance(user['created_at'], str):
            user['created_at'] = datetime.fromisoformat(user['created_at'])
        
        return User(**user)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ==================== FACE PROCESSING ====================

def generate_thumbnail(image_bytes: bytes, size=(300, 300)) -> bytes:
    """Generate thumbnail from image bytes and return thumbnail bytes"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Save to bytes
        output = io.BytesIO()
        img.save(output, "JPEG", quality=85)
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        logging.error(f"Error generating thumbnail: {e}")
        return image_bytes  # Return original if thumbnail fails

async def process_photo_faces(photo_id: str, event_id: str, gridfs_id: str):
    """Detect faces and generate embeddings for a photo. One embedding per face crop.
    OPTIMIZATION 4: Wraps DeepFace calls in thread pool to avoid blocking the event loop."""
    try:
        # Download photo from GridFS
        image_bytes = await download_from_gridfs(photos_fs, gridfs_id)
        
        # Save to temp file for DeepFace (DeepFace requires file paths)
        temp_image_path = TEMP_DIR / f"temp_{uuid.uuid4()}.jpg"
        with open(temp_image_path, 'wb') as f:
            f.write(image_bytes)
        
        try:
            # RetinaFace finds ALL faces in group photos; opencv often finds only the most prominent
            # OPTIMIZATION 4: Run DeepFace in thread pool to avoid blocking
            # Use run_in_executor for Python 3.7+ compatibility (to_thread requires 3.9+)
            loop = asyncio.get_event_loop()
            faces = await loop.run_in_executor(
                None,
                lambda: DeepFace.extract_faces(
                    img_path=str(temp_image_path),
                    detector_backend=FACE_DETECTOR,
                    enforce_detection=False
                )
            )
            
            if not faces:
                logging.info(f"No faces detected in photo {photo_id}")
                return []
            
            logging.info(f"Photo {photo_id}: detected {len(faces)} face(s)")
            face_embeddings = []
            
            for idx, face_data in enumerate(faces):
                if face_data.get('confidence', 0) < 0.4:
                    continue
                
                face_region = face_data['facial_area']
                # Use the cropped face image so we get one embedding per face (not the full image)
                face_crop = face_data.get('face')
                if face_crop is None:
                    # Fallback: crop using facial_area and re-read (BGR 0-255 from cv2)
                    img = cv2.imread(str(temp_image_path))
                    if img is None:
                        continue
                    x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                    face_crop = img[max(0, y):y + h, max(0, x):x + w]
                    face_to_save = face_crop
                else:
                    # DeepFace extract_faces returns RGB, 0-1 float; cv2.imwrite expects BGR 0-255
                    if face_crop.max() <= 1.0:
                        face_crop = (np.clip(face_crop, 0, 1) * 255).astype(np.uint8)
                    if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                        face_to_save = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                    else:
                        face_to_save = face_crop
                
                # DeepFace.represent expects a file path; write crop to temp file
                crop_path = TEMP_DIR / f"face_crop_{uuid.uuid4()}.jpg"
                try:
                    cv2.imwrite(str(crop_path), face_to_save)
                    # OPTIMIZATION 4: Run DeepFace.represent in thread pool
                    # Use run_in_executor for Python 3.7+ compatibility
                    loop = asyncio.get_event_loop()
                    embedding_result = await loop.run_in_executor(
                        None,
                        lambda: DeepFace.represent(
                            img_path=str(crop_path),
                            model_name=FACE_MODEL,
                            detector_backend='skip',
                            enforce_detection=False
                        )
                    )
                finally:
                    if crop_path.exists():
                        crop_path.unlink()
                
                if embedding_result:
                    embedding = np.array(embedding_result[0]['embedding']).astype('float32')
                    # L2-normalize so similarity = 1 - d²/2 works (Facenet/Face models may not normalize)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    face_id = str(uuid.uuid4())
                    bbox = [face_region['x'], face_region['y'], face_region['w'], face_region['h']]
                    
                    face_doc = {
                        'id': face_id,
                        'photo_id': photo_id,
                        'event_id': event_id,
                        'face_id': face_id,
                        'bbox': bbox,
                        'embedding': embedding.tolist(),
                        'created_at': datetime.now(timezone.utc).isoformat()
                    }
                    
                    await db.face_embeddings.insert_one(face_doc)
                    face_embeddings.append((face_id, embedding))
            
            return face_embeddings
            
        finally:
            # Clean up temp file
            if temp_image_path.exists():
                temp_image_path.unlink()
        
    except Exception as e:
        logging.error(f"Error processing photo faces: {e}")
        return []

async def build_faiss_index(event_id: str):
    """Build FAISS index for an event and store in GridFS.
    OPTIMIZATION 2: Uses IndexHNSWFlat for ~10-50x faster search with minimal recall loss."""
    try:
        logger.info(f"🔨 Building FAISS index for event {event_id}...")
        
        # Get all face embeddings for this event
        embeddings_cursor = db.face_embeddings.find({"event_id": event_id}, {"_id": 0})
        embeddings_list = await embeddings_cursor.to_list(None)
        
        if not embeddings_list:
            logger.info(f"No embeddings found for event {event_id}")
            return
        
        logger.info(f"Found {len(embeddings_list)} face embeddings to index")
        
        # Extract embeddings and face IDs
        face_ids = []
        embeddings = []
        photo_map = {}  # face_id -> photo_id
        
        for emb_doc in embeddings_list:
            face_ids.append(emb_doc['face_id'])
            emb = np.array(emb_doc['embedding']).astype('float32')
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
            photo_map[emb_doc['face_id']] = emb_doc['photo_id']
        
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # Run FAISS index building in thread pool (CPU-intensive, blocks event loop)
        loop = asyncio.get_event_loop()
        index, index_data = await loop.run_in_executor(
            None,
            _build_faiss_index_sync,
            embeddings_array,
            dimension,
            len(embeddings)
        )
        
        # Store in memory
        faiss_indices[event_id] = index
        face_mappings[event_id] = {
            'face_ids': face_ids,
            'photo_map': photo_map
        }
        
        # Serialize mapping
        mapping_bytes = pickle.dumps({'face_ids': face_ids, 'photo_map': photo_map})
        
        # Store in GridFS
        index_filename = f"{event_id}.index"
        mapping_filename = f"{event_id}.pkl"
        
        logger.info(f"Uploading FAISS index to GridFS...")
        
        # Delete old index if exists
        async for grid_file in faiss_fs.find({"filename": index_filename}):
            await faiss_fs.delete(grid_file._id)
        async for grid_file in faiss_fs.find({"filename": mapping_filename}):
            await faiss_fs.delete(grid_file._id)
        
        # Upload new index and mapping
        await upload_to_gridfs(faiss_fs, index_data, index_filename)
        await upload_to_gridfs(faiss_fs, mapping_bytes, mapping_filename)
        
        logger.info(f"✅ FAISS index built successfully - {len(embeddings)} faces indexed")
        
    except Exception as e:
        logger.error(f"❌ Error building FAISS index: {e}", exc_info=True)

def _build_faiss_index_sync(embeddings_array, dimension, num_embeddings):
    """Synchronous FAISS index building (runs in thread pool)"""
    # OPTIMIZATION 2: Create FAISS HNSW index for faster search
    if num_embeddings > 100:
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 64
        logging.info(f"Building HNSW index with {num_embeddings} faces")
    else:
        index = faiss.IndexFlatL2(dimension)
        logging.info(f"Building Flat index with {num_embeddings} faces")
    
    # Add embeddings to index
    index.add(embeddings_array)
    
    # Serialize index to bytes
    index_data = faiss.serialize_index(index)
    
    return index, index_data

async def search_faces(event_id: str, query_embedding: np.ndarray, k: int = None) -> List[SearchResult]:
    """Search for similar faces in an event with threshold filtering. Returns all unique photos where the face appears (group + individual).
    OPTIMIZATION 3: Uses batch MongoDB $in lookup to eliminate N+1 query problem."""
    try:
        # Load index if not in memory
        if event_id not in faiss_indices:
            index_filename = f"{event_id}.index"
            mapping_filename = f"{event_id}.pkl"
            
            # Try to load from GridFS
            try:
                async for grid_file in faiss_fs.find({"filename": index_filename}).limit(1):
                    index_data = await download_from_gridfs(faiss_fs, str(grid_file._id))
                    index = faiss.deserialize_index(index_data)
                    faiss_indices[event_id] = index
                    break
                else:
                    logging.warning(f"FAISS index not found for event {event_id}")
                    return []
                
                async for grid_file in faiss_fs.find({"filename": mapping_filename}).limit(1):
                    mapping_data = await download_from_gridfs(faiss_fs, str(grid_file._id))
                    face_mappings[event_id] = pickle.loads(mapping_data)
                    break
                    
            except Exception as e:
                logging.warning(f"Error loading FAISS index from GridFS: {e}")
                return []
        
        index = faiss_indices[event_id]
        mapping = face_mappings[event_id]
        n_vectors = index.ntotal
        if n_vectors == 0:
            logging.warning(f"FAISS index for event {event_id} is empty.")
            return []
        
        # Search the ENTIRE index so we never miss same person in other photos (group/individual)
        k = n_vectors if (k is None or k > n_vectors) else k
        query_array = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_array, k)
        
        # Log for debugging
        top_d = distances[0][: min(5, len(distances[0]))]
        top_similarities = [float(1.0 - (d / 2.0)) for d in top_d]
        logging.info(f"Event {event_id} selfie search: index has {n_vectors} faces, top5 similarities={[round(s, 3) for s in top_similarities]}, threshold={MINIMUM_MATCH_THRESHOLD}")
        
        # Per-photo BEST similarity: a photo is included if ANY face in it meets threshold (use best score for that photo)
        photo_best_similarity = {}  # photo_id -> best similarity seen for that photo
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            face_id = mapping['face_ids'][idx]
            photo_id = mapping['photo_map'][face_id]
            similarity_score = float(1.0 - (distance / 2.0))
            if photo_id not in photo_best_similarity or similarity_score > photo_best_similarity[photo_id]:
                photo_best_similarity[photo_id] = similarity_score
        
        # Filter photos by threshold
        matching_photo_ids = [
            photo_id for photo_id, sim in photo_best_similarity.items()
            if sim >= MINIMUM_MATCH_THRESHOLD
        ]
        
        if not matching_photo_ids:
            logging.info(f"Event {event_id} selfie search: No photos above threshold")
            return []
        
        # OPTIMIZATION 3: Batch MongoDB lookup using $in operator
        # This eliminates N+1 queries (was: N find_one calls, now: 1 find call)
        photos_cursor = db.photos.find(
            {"id": {"$in": matching_photo_ids}},
            {"_id": 0}
        )
        photos_list = await photos_cursor.to_list(None)
        
        # Create photo lookup dict
        photos_dict = {photo['id']: photo for photo in photos_list}
        
        # Build results using pre-fetched photos
        results = []
        for photo_id in matching_photo_ids:
            photo = photos_dict.get(photo_id)
            if photo:
                results.append(SearchResult(
                    photo_id=photo_id,
                    event_id=event_id,
                    similarity_score=photo_best_similarity[photo_id],
                    thumbnail_url=f"/api/photos/{photo_id}/thumbnail",
                    photo_url=f"/api/photos/{photo_id}"
                ))
        
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        logging.info(f"Event {event_id} selfie search: {len(results)} photos above threshold (eliminated {len(matching_photo_ids) - len(results)} missing photos)")
        return results
        
    except Exception as e:
        logging.error(f"Error searching faces: {e}")
        return []

# ==================== API ROUTES ====================

@api_router.post("/auth/register", response_model=dict)
async def register(user: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    user_doc = {
        'id': user_id,
        'email': user.email,
        'password_hash': hash_password(user.password),
        'name': user.name,
        'role': user.role,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(user_doc)
    
    token = create_access_token({"sub": user_id})
    
    return {
        'token': token,
        'user': {
            'id': user_id,
            'email': user.email,
            'name': user.name,
            'role': user.role
        }
    }

@api_router.post("/auth/login", response_model=dict)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(credentials.password, user['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user['id']})
    
    return {
        'token': token,
        'user': {
            'id': user['id'],
            'email': user['email'],
            'name': user['name'],
            'role': user['role']
        }
    }

@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@api_router.post("/events", response_model=Event)
async def create_event(event: EventCreate, current_user: User = Depends(get_current_user)):
    if current_user.role != "organizer":
        raise HTTPException(status_code=403, detail="Only organizers can create events")
    
    event_id = str(uuid.uuid4())
    # Generate 8-character code using only uppercase letters (A-Z)
    # More user-friendly: no special characters, easy to type
    event_code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(8))
    
    event_doc = {
        'id': event_id,
        'organizer_id': current_user.id,
        'title': event.title,
        'description': event.description,
        'event_code': event_code,
        'date': event.date,
        'status': 'active',
        'total_photos': 0,
        'processed_photos': 0,
        'faces_detected': 0,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    await db.events.insert_one(event_doc)
    
    event_doc['created_at'] = datetime.fromisoformat(event_doc['created_at'])
    return Event(**event_doc)

@api_router.get("/events", response_model=List[Event])
async def get_events(current_user: User = Depends(get_current_user)):
    query = {"organizer_id": current_user.id} if current_user.role == "organizer" else {}
    events = await db.events.find(query, {"_id": 0}).to_list(1000)
    
    for event in events:
        if isinstance(event['created_at'], str):
            event['created_at'] = datetime.fromisoformat(event['created_at'])
    
    return events

@api_router.get("/events/{event_id}", response_model=Event)
async def get_event(event_id: str, current_user: User = Depends(get_current_user)):
    event = await db.events.find_one({"id": event_id}, {"_id": 0})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    if isinstance(event['created_at'], str):
        event['created_at'] = datetime.fromisoformat(event['created_at'])
    
    return Event(**event)

@api_router.get("/events/code/{event_code}", response_model=Event)
async def get_event_by_code(event_code: str):
    event = await db.events.find_one({"event_code": event_code.upper()}, {"_id": 0})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    if isinstance(event['created_at'], str):
        event['created_at'] = datetime.fromisoformat(event['created_at'])
    
    return Event(**event)

@api_router.post("/events/{event_id}/upload")
async def upload_photos(
    event_id: str,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
):
    event = await db.events.find_one({"id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    if event['organizer_id'] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    uploaded_photos = []
    
    for file in files:
        photo_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1]
        photo_filename = f"{photo_id}.{file_extension}"
        
        # Read file content
        content = await file.read()
        
        # Upload photo to GridFS
        gridfs_id = await upload_to_gridfs(
            photos_fs, 
            content, 
            photo_filename,
            metadata={"event_id": event_id, "photo_id": photo_id}
        )
        
        # Generate and upload thumbnail to GridFS
        thumbnail_bytes = generate_thumbnail(content)
        thumbnail_filename = f"{photo_id}_thumb.jpg"
        thumbnail_gridfs_id = await upload_to_gridfs(
            thumbnails_fs,
            thumbnail_bytes,
            thumbnail_filename,
            metadata={"event_id": event_id, "photo_id": photo_id}
        )
        
        # Save to database with GridFS IDs
        photo_doc = {
            'id': photo_id,
            'event_id': event_id,
            'gridfs_id': gridfs_id,
            'thumbnail_gridfs_id': thumbnail_gridfs_id,
            'upload_status': 'uploaded',
            'processed': False,
            'faces_detected': 0,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        await db.photos.insert_one(photo_doc)
        uploaded_photos.append(photo_id)
        logging.info(f"Uploaded photo {photo_id} to GridFS")
    
    # Update event stats
    await db.events.update_one(
        {"id": event_id},
        {"$inc": {"total_photos": len(files)}}
    )
    
    # Start processing in background
    asyncio.create_task(process_event_photos(event_id))
    
    return {
        'message': f'{len(files)} photos uploaded successfully',
        'photo_ids': uploaded_photos
    }

@api_router.post("/events/{event_id}/reprocess")
async def reprocess_event(event_id: str, current_user: User = Depends(get_current_user)):
    """Re-extract faces from all photos and rebuild FAISS index (fixes matching without re-uploading)."""
    event = await db.events.find_one({"id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if event['organizer_id'] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to reprocess this event")
    
    # Remove old face embeddings
    await db.face_embeddings.delete_many({"event_id": event_id})
    
    # Remove old FAISS index from GridFS
    index_filename = f"{event_id}.index"
    mapping_filename = f"{event_id}.pkl"
    async for grid_file in faiss_fs.find({"filename": index_filename}):
        await faiss_fs.delete(grid_file._id)
    async for grid_file in faiss_fs.find({"filename": mapping_filename}):
        await faiss_fs.delete(grid_file._id)
    
    # Clear from memory
    if event_id in faiss_indices:
        del faiss_indices[event_id]
    if event_id in face_mappings:
        del face_mappings[event_id]
    
    # Mark all photos as unprocessed
    await db.photos.update_many(
        {"event_id": event_id},
        {"$set": {"processed": False, "upload_status": "uploaded", "faces_detected": 0}}
    )
    await db.events.update_one(
        {"id": event_id},
        {"$set": {"status": "processing", "processed_photos": 0, "faces_detected": 0}}
    )
    
    asyncio.create_task(process_event_photos(event_id))
    return {"message": "Reprocessing started. Face index will rebuild; try selfie search again when status is completed.", "event_id": event_id}

@api_router.delete("/events/{event_id}")
async def delete_event(event_id: str, current_user: User = Depends(get_current_user)):
    """Delete an event and all associated data (photos, embeddings, FAISS index)"""
    # Verify event exists and user has permission
    event = await db.events.find_one({"id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    if event['organizer_id'] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this event")
    
    try:
        logging.info(f"Deleting event {event_id} and all associated data")
        
        # 1. Get all photos for this event
        photos = await db.photos.find({"event_id": event_id}, {"_id": 0}).to_list(None)
        logging.info(f"Found {len(photos)} photos to delete")
        
        # 2. Delete all photos and thumbnails from GridFS
        for photo in photos:
            try:
                await delete_from_gridfs(photos_fs, photo['gridfs_id'])
                await delete_from_gridfs(thumbnails_fs, photo['thumbnail_gridfs_id'])
            except Exception as e:
                logging.warning(f"Error deleting photo {photo['id']} from GridFS: {e}")
        
        # 3. Delete all photos from database
        result = await db.photos.delete_many({"event_id": event_id})
        logging.info(f"Deleted {result.deleted_count} photo records")
        
        # 4. Delete all face embeddings
        result = await db.face_embeddings.delete_many({"event_id": event_id})
        logging.info(f"Deleted {result.deleted_count} face embeddings")
        
        # 5. Delete FAISS index from GridFS
        index_filename = f"{event_id}.index"
        mapping_filename = f"{event_id}.pkl"
        async for grid_file in faiss_fs.find({"filename": index_filename}):
            await faiss_fs.delete(grid_file._id)
            logging.info(f"Deleted FAISS index from GridFS")
        async for grid_file in faiss_fs.find({"filename": mapping_filename}):
            await faiss_fs.delete(grid_file._id)
            logging.info(f"Deleted FAISS mapping from GridFS")
        
        # 6. Clear from memory cache
        if event_id in faiss_indices:
            del faiss_indices[event_id]
        if event_id in face_mappings:
            del face_mappings[event_id]
        
        # 7. Delete the event itself
        await db.events.delete_one({"id": event_id})
        logging.info(f"Deleted event {event_id}")
        
        return {
            "message": "Event and all associated data deleted successfully",
            "event_id": event_id,
            "photos_deleted": len(photos)
        }
        
    except Exception as e:
        logging.error(f"Error deleting event {event_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete event: {str(e)}")

async def process_event_photos(event_id: str):
    """Background task to process all photos in an event"""
    try:
        logger.info(f"🎬 Starting photo processing for event {event_id}")
        
        # Update event status
        await db.events.update_one(
            {"id": event_id},
            {"$set": {"status": "processing"}}
        )
        
        # Get all unprocessed photos
        photos = await db.photos.find(
            {"event_id": event_id, "processed": False},
            {"_id": 0}
        ).to_list(None)
        
        total = len(photos)
        logger.info(f"📸 Found {total} unprocessed photos for event {event_id}")
        
        if total == 0:
            logger.info(f"✅ No photos to process for event {event_id}")
            await db.events.update_one(
                {"id": event_id},
                {"$set": {"status": "completed"}}
            )
            return
        
        for idx, photo in enumerate(photos):
            try:
                progress = f"[{idx + 1}/{total}]"
                logger.info(f"{progress} 🔍 Processing photo {photo['id']}")
                
                # Process faces using GridFS ID
                face_embeddings = await process_photo_faces(
                    photo['id'],
                    event_id,
                    photo['gridfs_id']
                )
                
                logger.info(f"{progress} ✓ Photo {photo['id']}: detected {len(face_embeddings)} face(s)")
                
                # Update photo
                await db.photos.update_one(
                    {"id": photo['id']},
                    {
                        "$set": {
                            "processed": True,
                            "faces_detected": len(face_embeddings),
                            "upload_status": "completed"
                        }
                    }
                )
                
                # Update event stats
                await db.events.update_one(
                    {"id": event_id},
                    {
                        "$inc": {
                            "processed_photos": 1,
                            "faces_detected": len(face_embeddings)
                        }
                    }
                )
                
            except Exception as e:
                logger.error(f"❌ Error processing photo {photo['id']}: {e}", exc_info=True)
                await db.photos.update_one(
                    {"id": photo['id']},
                    {"$set": {"upload_status": "failed"}}
                )
        
        # Build FAISS index
        logger.info(f"🔨 Building FAISS index for event {event_id}")
        await build_faiss_index(event_id)
        
        # Update event status
        await db.events.update_one(
            {"id": event_id},
            {"$set": {"status": "completed"}}
        )
        
        logger.info(f"✅ Completed processing for event {event_id} - {total} photos processed")
        
    except Exception as e:
        logger.error(f"❌ Error processing event photos for {event_id}: {e}", exc_info=True)
        await db.events.update_one(
            {"id": event_id},
            {"$set": {"status": "failed"}}
        )

@api_router.get("/events/{event_id}/status", response_model=ProcessingStatus)
async def get_processing_status(event_id: str, current_user: User = Depends(get_current_user)):
    event = await db.events.find_one({"id": event_id}, {"_id": 0})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    return ProcessingStatus(
        event_id=event_id,
        total_photos=event['total_photos'],
        processed_photos=event['processed_photos'],
        faces_detected=event['faces_detected'],
        status=event['status']
    )

@api_router.get("/events/{event_id}/photos", response_model=List[Photo])
async def get_event_photos(event_id: str, current_user: User = Depends(get_current_user)):
    photos = await db.photos.find({"event_id": event_id}, {"_id": 0}).to_list(1000)
    
    for photo in photos:
        if isinstance(photo['created_at'], str):
            photo['created_at'] = datetime.fromisoformat(photo['created_at'])
    
    return photos

@api_router.post("/search/selfie", response_model=List[SearchResult])
async def search_by_selfie(
    event_id: str = Form(...),
    file: UploadFile = File(...)
):
    # Verify event exists
    event = await db.events.find_one({"id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    if event['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Event processing not completed")
    
    try:
        # Read selfie bytes for caching
        content = await file.read()
        
        # OPTIMIZATION 1: Check if we've already computed embedding for this selfie
        selfie_hash = hash_image_bytes(content)
        
        if selfie_hash in selfie_embedding_cache:
            logging.info(f"Cache HIT: Using cached embedding for selfie hash {selfie_hash[:16]}...")
            query_embedding = selfie_embedding_cache[selfie_hash]
        else:
            logging.info(f"Cache MISS: Computing new embedding for selfie hash {selfie_hash[:16]}...")
            
            # Save temporary selfie to temp directory
            selfie_path = TEMP_DIR / f"temp_{uuid.uuid4()}.jpg"
            with open(selfie_path, 'wb') as f:
                f.write(content)
            
            try:
                # OPTIMIZATION 4: Run DeepFace in thread pool to avoid blocking
                # Use run_in_executor for Python 3.7+ compatibility
                loop = asyncio.get_event_loop()
                embedding_result = await loop.run_in_executor(
                    None,
                    lambda: DeepFace.represent(
                        img_path=str(selfie_path),
                        model_name=FACE_MODEL,
                        detector_backend=FACE_DETECTOR,
                        enforce_detection=False
                    )
                )
                
                if not embedding_result:
                    raise HTTPException(status_code=400, detail="No face detected in selfie")
                
                query_embedding = np.array(embedding_result[0]['embedding']).astype('float32')
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
                
                # OPTIMIZATION 1: Store in cache for future requests
                selfie_embedding_cache[selfie_hash] = query_embedding
                logging.info(f"Cached embedding for selfie {selfie_hash[:16]}... (cache size: {len(selfie_embedding_cache)})")
                
            finally:
                # Clean up temp file
                if selfie_path.exists():
                    selfie_path.unlink()
        
        # Search for similar faces
        results = await search_faces(event_id, query_embedding)
        
        # If no photos meet the minimum threshold, return appropriate message
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="No matching photos found. You are not present in this event."
            )
        
        return results
        
    except HTTPException:
        # Re-raise HTTP exceptions (including our custom 404)
        raise
    except Exception as e:
        logging.error(f"Error searching by selfie: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/photos/{photo_id}")
async def get_photo(photo_id: str):
    photo = await db.photos.find_one({"id": photo_id}, {"_id": 0})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    # Download from GridFS
    image_data = await download_from_gridfs(photos_fs, photo['gridfs_id'])
    
    return Response(content=image_data, media_type="image/jpeg")

@api_router.get("/photos/{photo_id}/thumbnail")
async def get_photo_thumbnail(photo_id: str):
    photo = await db.photos.find_one({"id": photo_id}, {"_id": 0})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    # Download from GridFS
    thumbnail_data = await download_from_gridfs(thumbnails_fs, photo['thumbnail_gridfs_id'])
    
    return Response(content=thumbnail_data, media_type="image/jpeg")

@api_router.delete("/events/{event_id}/photos/{photo_id}")
async def delete_photo(
    event_id: str, 
    photo_id: str, 
    current_user: User = Depends(get_current_user)
):
    """Delete a photo completely from all storage systems (GridFS + Database)"""
    # Verify event exists and user has permission
    event = await db.events.find_one({"id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    if event['organizer_id'] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete photos from this event")
    
    # Verify photo exists
    photo = await db.photos.find_one({"id": photo_id, "event_id": event_id})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    try:
        # 1. Remove from GridFS
        await delete_from_gridfs(photos_fs, photo['gridfs_id'])
        await delete_from_gridfs(thumbnails_fs, photo['thumbnail_gridfs_id'])
        logging.info(f"Deleted photo {photo_id} from GridFS")
        
        # 2. Remove from database
        await db.photos.delete_one({"id": photo_id})
        
        # 3. Remove face embeddings from database
        await db.face_embeddings.delete_many({"photo_id": photo_id})
        
        # 4. Update event statistics
        await db.events.update_one(
            {"id": event_id},
            {
                "$inc": {
                    "total_photos": -1,
                    "processed_photos": -1 if photo['processed'] else 0,
                    "faces_detected": -photo['faces_detected']
                }
            }
        )
        
        # 5. Rebuild FAISS index for the event (if index exists)
        try:
            await rebuild_faiss_index(event_id)
        except Exception as e:
            logging.warning(f"Failed to rebuild FAISS index after deletion: {e}")
            # Don't fail the deletion if index rebuild fails
        
        return {
            "message": "Photo deleted successfully",
            "photo_id": photo_id,
            "event_id": event_id
        }
        
    except Exception as e:
        logging.error(f"Error deleting photo {photo_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete photo")

async def rebuild_faiss_index(event_id: str):
    """Rebuild FAISS index after photo deletion"""
    try:
        # Remove existing index from GridFS
        index_filename = f"{event_id}.index"
        mapping_filename = f"{event_id}.pkl"
        async for grid_file in faiss_fs.find({"filename": index_filename}):
            await faiss_fs.delete(grid_file._id)
        async for grid_file in faiss_fs.find({"filename": mapping_filename}):
            await faiss_fs.delete(grid_file._id)
        
        # Remove from memory
        if event_id in faiss_indices:
            del faiss_indices[event_id]
        if event_id in face_mappings:
            del face_mappings[event_id]
        
        # Rebuild index with remaining embeddings
        await build_faiss_index(event_id)
        
    except Exception as e:
        logging.error(f"Error rebuilding FAISS index for event {event_id}: {e}")
        raise

@api_router.get("/")
async def root():
    return {"message": "FaceShot API"}

# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporarily disable startup event for debugging
# @app.on_event("startup")
# async def startup_event():
#     """Warm up models on server start to avoid first-request delays"""
#     logger.info("🚀 Starting Event Snap server...")
#     logger.info("📦 MongoDB connected")
#     
#     # Skip model warmup for now (can cause startup issues)
#     # Models will load on first photo upload instead
#     logger.info("⚡ Server ready! (Models will load on first use)")
#     
#     # Optionally run warmup in background (non-blocking, won't crash server)
#     # Uncomment to enable:
#     # try:
#     #     loop = asyncio.get_event_loop()
#     #     loop.run_in_executor(None, warmup_models)
#     # except Exception as e:
#     #     logger.warning(f"Model warmup failed (non-critical): {e}")

def warmup_models():
    """Pre-load DeepFace models"""
    try:
        import numpy as np
        import cv2
        
        # Create dummy image
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        temp_path = TEMP_DIR / "warmup.jpg"
        cv2.imwrite(str(temp_path), dummy)
        
        try:
            # Load models
            DeepFace.extract_faces(str(temp_path), detector_backend=FACE_DETECTOR, enforce_detection=False)
            DeepFace.represent(str(temp_path), model_name=FACE_MODEL, detector_backend='skip', enforce_detection=False)
            logger.info("✓ Models loaded and ready")
        finally:
            if temp_path.exists():
                temp_path.unlink()
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    logger.info("Shutting down Event Snap server...")
    client.close()


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Use port 8081 (bypasses Windows Hyper-V 7910-8009 exclusion range)
    PORT = int(os.environ.get("PORT", 8081))
    
    print("="*60)
    print("EVENT SNAP BACKEND SERVER")
    print("="*60)
    print(f"Running on: http://localhost:{PORT}")
    print(f"API Docs:   http://localhost:{PORT}/docs")
    print(f"API Root:   http://localhost:{PORT}/api/")
    print("")
    print("Press CTRL+C to stop")
    print("="*60)
    print("")
    
    try:
        uvicorn.run(
            "server:app",
            host="127.0.0.1",
            port=PORT,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nServer error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
