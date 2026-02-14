"""
Test script to verify the updated face matching logic with thresholds
"""

import numpy as np
from dataclasses import dataclass
from typing import List

# Simulate the threshold constants
MINIMUM_MATCH_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.7

@dataclass
class SearchResult:
    photo_id: str
    event_id: str
    similarity_score: float
    thumbnail_url: str
    photo_url: str

def test_similarity_scenarios():
    """Test various similarity score scenarios"""
    
    print("=== Face Matching Logic Test ===\n")
    
    # Test cases with different similarity scores
    test_cases = [
        # No matches scenario
        {
            'name': 'No Matches',
            'scores': [0.3, 0.4, 0.45],  # All below minimum threshold
            'expected_result': 'NO_MATCHES',
            'expected_message': 'No matching photos found. You are not present in this event.'
        },
        # Low confidence matches only
        {
            'name': 'Low Confidence Matches',
            'scores': [0.55, 0.6, 0.52],  # Above minimum, below high confidence
            'expected_result': 'POSSIBLE_MATCHES',
            'expected_message': 'Found matches with possible confidence'
        },
        # High confidence matches
        {
            'name': 'High Confidence Matches',
            'scores': [0.75, 0.8, 0.72],  # Above high confidence threshold
            'expected_result': 'HIGH_CONFIDENCE_MATCHES',
            'expected_message': 'Found matches with high confidence'
        },
        # Mixed confidence matches
        {
            'name': 'Mixed Confidence Matches',
            'scores': [0.4, 0.65, 0.75, 0.3],  # Mixed - some above minimum, some below
            'expected_result': 'MIXED_MATCHES',
            'expected_message': 'Found matches with varying confidence levels'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 40)
        
        # Simulate search results
        results = []
        for j, score in enumerate(test_case['scores']):
            results.append(SearchResult(
                photo_id=f"photo_{j+1}",
                event_id="test_event",
                similarity_score=score,
                thumbnail_url=f"/api/photos/photo_{j+1}/thumbnail",
                photo_url=f"/api/photos/photo_{j+1}"
            ))
        
        # Apply filtering (same logic as backend)
        filtered_results = [r for r in results if r.similarity_score >= MINIMUM_MATCH_THRESHOLD]
        filtered_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Determine result type
        if not filtered_results:
            result_type = "NO_MATCHES"
            print("✓ No matches found (below minimum threshold)")
            print(f"  Message: {test_case['expected_message']}")
        else:
            high_confidence_count = sum(1 for r in filtered_results if r.similarity_score >= HIGH_CONFIDENCE_THRESHOLD)
            low_confidence_count = len(filtered_results) - high_confidence_count
            
            print(f"✓ Found {len(filtered_results)} matching photos:")
            for result in filtered_results:
                confidence_label = "High Confidence" if result.similarity_score >= HIGH_CONFIDENCE_THRESHOLD else "Possible Match"
                print(f"  - Photo {result.photo_id}: {result.similarity_score:.1%} ({confidence_label})")
            
            if high_confidence_count > 0 and low_confidence_count > 0:
                result_type = "MIXED_MATCHES"
            elif high_confidence_count > 0:
                result_type = "HIGH_CONFIDENCE_MATCHES"
            else:
                result_type = "POSSIBLE_MATCHES"
        
        print(f"  Result Type: {result_type}")
        print(f"  Expected: {test_case['expected_result']}")
        print(f"  Match: {'✓' if result_type == test_case['expected_result'] else '✗'}")
        print()

def test_threshold_boundaries():
    """Test edge cases around threshold boundaries"""
    
    print("=== Threshold Boundary Tests ===\n")
    
    boundary_tests = [
        (MINIMUM_MATCH_THRESHOLD - 0.01, "Below minimum threshold - should be filtered out"),
        (MINIMUM_MATCH_THRESHOLD, "At minimum threshold - should be included"),
        (MINIMUM_MATCH_THRESHOLD + 0.01, "Above minimum threshold - should be included"),
        (HIGH_CONFIDENCE_THRESHOLD - 0.01, "Below high confidence - labeled as 'Possible Match'"),
        (HIGH_CONFIDENCE_THRESHOLD, "At high confidence - labeled as 'High Confidence'"),
        (HIGH_CONFIDENCE_THRESHOLD + 0.01, "Above high confidence - labeled as 'High Confidence'")
    ]
    
    for score, description in boundary_tests:
        meets_minimum = score >= MINIMUM_MATCH_THRESHOLD
        is_high_confidence = score >= HIGH_CONFIDENCE_THRESHOLD
        label = "High Confidence" if is_high_confidence else "Possible Match" if meets_minimum else "Filtered Out"
        
        print(f"Score: {score:.2f} ({score*100:.0f}%)")
        print(f"  Description: {description}")
        print(f"  Meets minimum threshold: {meets_minimum}")
        print(f"  Is high confidence: {is_high_confidence}")
        print(f"  Label: {label}")
        print()

if __name__ == "__main__":
    test_similarity_scenarios()
    test_threshold_boundaries()
    
    print("=== Configuration Summary ===")
    print(f"Minimum Match Threshold: {MINIMUM_MATCH_THRESHOLD:.0%}")
    print(f"High Confidence Threshold: {HIGH_CONFIDENCE_THRESHOLD:.0%}")
    print("\nAll tests completed!")