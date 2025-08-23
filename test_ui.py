#!/usr/bin/env python3

import requests
import json
import time

def test_movie_benchmark_ui():
    """Test the Movie Benchmark UI endpoints."""
    base_url = "http://localhost:8002"
    
    print("🧪 Testing Movie Vector Benchmark UI")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Database availability
    print("\n2. Testing database availability...")
    try:
        response = requests.get(f"{base_url}/api/databases")
        if response.status_code == 200:
            databases = response.json()
            print("✅ Database availability check passed")
            print("   Available databases:")
            for name, info in databases.items():
                status = "🟢" if info['available'] else "🔴"
                print(f"   {status} {info['name']} - {info.get('description', '')}")
        else:
            print(f"❌ Database availability failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Database availability failed: {e}")
        return
    
    # Test 3: Initialize system
    print("\n3. Testing system initialization...")
    try:
        response = requests.post(f"{base_url}/api/initialize")
        if response.status_code == 200:
            print("✅ System initialization started")
            
            # Poll for initialization completion
            max_attempts = 30
            for attempt in range(max_attempts):
                status_response = requests.get(f"{base_url}/api/status")
                status = status_response.json()
                
                print(f"   Status: {status['status']} - {status['message']} ({status['progress']*100:.1f}%)")
                
                if status['status'] == 'ready':
                    print("✅ System initialization completed!")
                    break
                elif status['status'] == 'error':
                    print(f"❌ System initialization failed: {status['message']}")
                    return
                
                time.sleep(2)
            else:
                print("❌ System initialization timed out")
                return
                
        else:
            print(f"❌ System initialization failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Test 4: Benchmark (if system is ready)
    print("\n4. Testing benchmark functionality...")
    available_dbs = ['faiss', 'chroma', 'qdrant']  # Our working databases
    
    # Add cloud databases if API keys are available
    import os
    if os.getenv('PINECONE_API_KEY'):
        available_dbs.append('pinecone')
        print("   Added Pinecone to test (API key found)")
    if os.getenv('TOPK_API_KEY'):
        available_dbs.append('topk')
        print("   Added TopK to test (API key found)")
    
    benchmark_request = {
        "databases": available_dbs,
        "sample_size": 100,  # Small sample for quick test
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/benchmark",
            json=benchmark_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Benchmark started successfully")
            
            # Poll for benchmark completion
            max_attempts = 30
            for attempt in range(max_attempts):
                results_response = requests.get(f"{base_url}/api/benchmark/results")
                results = results_response.json()
                
                print(f"   Status: {results['status']} - {results['message']} ({results.get('progress', 0)*100:.1f}%)")
                
                if results['status'] == 'completed' and 'results' in results:
                    print("✅ Benchmark completed successfully!")
                    print(f"   Tested {len(results['results'])} databases:")
                    
                    for result in results['results']:
                        print(f"     • {result['database'].upper()}: "
                              f"{result['throughput']} vec/s, "
                              f"{result['avg_latency']}ms latency, "
                              f"{result['recall_at_10']} recall@10")
                    
                    break
                elif results['status'] == 'error':
                    print(f"❌ Benchmark failed: {results['message']}")
                    return
                    
                time.sleep(5)
            else:
                print("❌ Benchmark timed out")
                return
                
        else:
            result = response.json()
            print(f"❌ Benchmark failed to start: {result.get('detail', response.status_code)}")
            return
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        return
    
    print("\n🎉 All tests passed! The Movie Vector Benchmark UI is working correctly.")
    print(f"\n🌐 Access the web interface at: {base_url}")
    print("\n📊 Features available:")
    print("   • Interactive movie search across all databases")
    print("   • Comprehensive benchmarking with 4+ vector databases")
    print("   • Real-time performance visualization")
    print("   • Detailed comparison charts and radar plots")

if __name__ == "__main__":
    test_movie_benchmark_ui()