#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced lookback functionality.
"""

import subprocess
import sys
import os

def test_lookback_functionality():
    """Test the new lookback features."""
    print("🧪 Testing Enhanced Lookback Functionality")
    print("=" * 50)
    
    # Test 1: Show historical context
    print("\n1. Testing historical context generation...")
    try:
        from forecast import _get_prediction_history
        context = _get_prediction_history(3)
        print(f"✅ Historical context generated ({len(context)} characters)")
        print("Sample context:")
        print(context[:500] + "..." if len(context) > 500 else context)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Show prediction updates functionality
    print("\n2. Testing prediction updates...")
    try:
        from forecast import _add_prediction_updates_to_newsletter
        today = "2025-07-27"  # Use a known date
        _add_prediction_updates_to_newsletter(today)
        print("✅ Prediction updates function executed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Show available commands
    print("\n3. Available new commands:")
    print("   python forecast.py predict          # Now includes historical context")
    print("   python forecast.py mark-outcome     # Mark prediction outcomes")
    print("   python forecast.py run-daily        # Now includes prediction updates")
    
    print("\n📝 Usage Examples:")
    print("   # Mark a prediction as correct:")
    print("   python forecast.py mark-outcome --prediction-index 0 --outcome correct")
    print("   ")
    print("   # Mark a prediction as incorrect:")
    print("   python forecast.py mark-outcome --prediction-index 1 --outcome incorrect")
    print("   ")
    print("   # Run full pipeline with lookbacks:")
    print("   python forecast.py run-daily")

if __name__ == "__main__":
    test_lookback_functionality() 