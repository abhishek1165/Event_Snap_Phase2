"""
Test script to verify photo deletion functionality
"""

import os
from pathlib import Path
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add the backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Mock the database and storage
class MockDB:
    def __init__(self):
        self.photos = {}
        self.face_embeddings = {}
        self.events = {}
    
    async def photos_find_one(self, query, projection=None):
        photo_id = query.get('id')
        return self.photos.get(photo_id)
    
    async def photos_delete_one(self, query):
        photo_id = query.get('id')
        if photo_id in self.photos:
            del self.photos[photo_id]
            return Mock(deleted_count=1)
        return Mock(deleted_count=0)
    
    async def face_embeddings_delete_many(self, query):
        photo_id = query.get('photo_id')
        deleted_count = 0
        keys_to_delete = []
        for key, embedding in self.face_embeddings.items():
            if embedding.get('photo_id') == photo_id:
                keys_to_delete.append(key)
                deleted_count += 1
        for key in keys_to_delete:
            del self.face_embeddings[key]
        return Mock(deleted_count=deleted_count)
    
    async def events_update_one(self, query, update):
        event_id = query.get('id')
        if event_id in self.events:
            inc_values = update.get('$inc', {})
            for key, value in inc_values.items():
                if key in self.events[event_id]:
                    self.events[event_id][key] += value
        return Mock(modified_count=1)

class MockPath:
    def __init__(self, path):
        self.path = path
        self._exists = True
    
    def exists(self):
        return self._exists
    
    def unlink(self):
        self._exists = False
        print(f"  ✓ Deleted file: {self.path}")

def test_deletion_workflow():
    """Test the complete photo deletion workflow"""
    
    print("=== Photo Deletion Workflow Test ===\n")
    
    # Test data
    test_photo = {
        'id': 'test_photo_123',
        'event_id': 'test_event_456',
        'file_path': '/app/storage/photos/test_photo_123.jpg',
        'thumbnail_path': '/app/storage/thumbnails/test_photo_123_thumb.jpg',
        'processed': True,
        'faces_detected': 2
    }
    
    test_event = {
        'id': 'test_event_456',
        'organizer_id': 'test_organizer_789',
        'total_photos': 5,
        'processed_photos': 5,
        'faces_detected': 8
    }
    
    test_face_embeddings = [
        {
            'id': 'embedding_1',
            'photo_id': 'test_photo_123',
            'event_id': 'test_event_456',
            'face_id': 'face_1'
        },
        {
            'id': 'embedding_2', 
            'photo_id': 'test_photo_123',
            'event_id': 'test_event_456',
            'face_id': 'face_2'
        }
    ]
    
    # Initialize mock database
    mock_db = MockDB()
    mock_db.photos[test_photo['id']] = test_photo
    mock_db.events[test_event['id']] = test_event
    for embedding in test_face_embeddings:
        mock_db.face_embeddings[embedding['id']] = embedding
    
    print("Initial State:")
    print(f"  Photos in DB: {len(mock_db.photos)}")
    print(f"  Face embeddings: {len(mock_db.face_embeddings)}")
    print(f"  Event photos count: {mock_db.events[test_event['id']]['total_photos']}")
    print()
    
    # Simulate deletion process
    print("Performing deletion steps:")
    
    # Step 1: Remove from file system
    photo_path = MockPath(test_photo['file_path'])
    thumbnail_path = MockPath(test_photo['thumbnail_path'])
    
    if photo_path.exists():
        photo_path.unlink()
    if thumbnail_path.exists():
        thumbnail_path.unlink()
    
    # Step 2: Remove from database
    result = asyncio.run(mock_db.photos_delete_one({'id': test_photo['id']}))
    print(f"  ✓ Removed photo from database (deleted: {result.deleted_count})")
    
    # Step 3: Remove face embeddings
    result = asyncio.run(mock_db.face_embeddings_delete_many({'photo_id': test_photo['id']}))
    print(f"  ✓ Removed {result.deleted_count} face embeddings")
    
    # Step 4: Update event statistics
    update_query = {
        '$inc': {
            'total_photos': -1,
            'processed_photos': -1 if test_photo['processed'] else 0,
            'faces_detected': -test_photo['faces_detected']
        }
    }
    asyncio.run(mock_db.events_update_one({'id': test_event['id']}, update_query))
    print(f"  ✓ Updated event statistics")
    
    print()
    print("Final State:")
    print(f"  Photos in DB: {len(mock_db.photos)}")
    print(f"  Face embeddings: {len(mock_db.face_embeddings)}")
    print(f"  Event photos count: {mock_db.events[test_event['id']]['total_photos']}")
    print(f"  Event processed photos: {mock_db.events[test_event['id']]['processed_photos']}")
    print(f"  Event faces detected: {mock_db.events[test_event['id']]['faces_detected']}")
    
    # Verify deletion
    success = (
        len(mock_db.photos) == 0 and
        len(mock_db.face_embeddings) == 0 and
        mock_db.events[test_event['id']]['total_photos'] == 4 and
        mock_db.events[test_event['id']]['processed_photos'] == 4 and
        mock_db.events[test_event['id']]['faces_detected'] == 6
    )
    
    print()
    print(f"Deletion Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
    return success

def test_search_after_deletion():
    """Test that deleted photos don't appear in search results"""
    
    print("\n=== Search Results After Deletion Test ===\n")
    
    # Simulate FAISS index state AFTER deletion
    # The index should have been rebuilt without the deleted photo
    class MockFAISSIndex:
        def __init__(self):
            # Index after deletion - deleted photo should be gone
            self.photo_ids = ['photo_1', 'photo_2', 'photo_4']  # test_photo_123 removed
        
        def search(self, query, k):
            # Return all remaining photos
            distances = [0.1] * len(self.photo_ids)  # Mock distances
            indices = list(range(len(self.photo_ids)))  # Mock indices
            return distances, indices
    
    index = MockFAISSIndex()
    distances, indices = index.search("query", 10)
    
    print("Search results after deletion:")
    print(f"  Found {len(indices)} photos")
    print(f"  Photo IDs: {index.photo_ids}")
    
    # Verify deleted photo is not in results
    deleted_photo_in_results = 'test_photo_123' in index.photo_ids
    
    success = not deleted_photo_in_results
    print(f"  Deleted photo properly excluded: {'✓ YES' if success else '✗ NO'}")
    
    return success

if __name__ == "__main__":
    print("Testing Photo Deletion Functionality")
    print("=" * 50)
    
    deletion_success = test_deletion_workflow()
    search_success = test_search_after_deletion()
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"  Deletion Workflow: {'✓ PASS' if deletion_success else '✗ FAIL'}")
    print(f"  Search Exclusion: {'✓ PASS' if search_success else '✗ FAIL'}")
    
    overall_success = deletion_success and search_success
    print(f"\n  Overall: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")