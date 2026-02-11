#!/usr/bin/env python3
"""
Test Questions for RAG Pipeline
A comprehensive set of test queries to validate the RAG pipeline
"""

# List of diverse test queries covering different aspects of ARGO float data
TEST_QUERIES = [
    # Basic queries
    {
        "id": 1,
        "query": "What are the floats in the Pacific Ocean?",
        "category": "Geographic",
        "expected_fields": ["latitude", "longitude", "ocean", "platform_number"]
    },
    
    {
        "id": 2,
        "query": "Show me floats with high temperature readings above 25 degrees Celsius",
        "category": "Temperature",
        "expected_fields": ["temp_adjusted_c", "platform_number", "latitude", "longitude"]
    },
    
    {
        "id": 3,
        "query": "Find floats near the equator",
        "category": "Geographic",
        "expected_fields": ["latitude", "longitude", "platform_number"]
    },
    
    # Biogeochemistry queries
    {
        "id": 4,
        "query": "Which floats measure dissolved oxygen levels in the ocean?",
        "category": "Biogeochemistry",
        "expected_fields": ["PROFILE_DOXY_QC", "float_type", "platform_number"]
    },
    
    {
        "id": 5,
        "query": "Show me floats with chlorophyll measurements",
        "category": "Biogeochemistry",
        "expected_fields": ["PROFILE_CHLA_QC", "float_type", "platform_number"]
    },
    
    {
        "id": 6,
        "query": "Find BCG floats with CDOM measurements",
        "category": "Biogeochemistry",
        "expected_fields": ["cdom_adjusted", "float_type", "platform_number"]
    },
    
    # Depth and salinity queries
    {
        "id": 7,
        "query": "What floats measure deep ocean waters below 2000 meters?",
        "category": "Depth",
        "expected_fields": ["max_depth_m", "platform_number", "latitude"]
    },
    
    {
        "id": 8,
        "query": "Find floats with high salinity readings above 35 PSU",
        "category": "Salinity",
        "expected_fields": ["psal_adjusted_psu", "platform_number", "latitude"]
    },
    
    # Quality and data mode queries
    {
        "id": 9,
        "query": "Show me delayed-mode high quality float data",
        "category": "DataQuality",
        "expected_fields": ["data_mode", "PROFILE_TEMP_QC", "platform_number"]
    },
    
    {
        "id": 10,
        "query": "Find floats in the Southern Ocean with good quality measurements",
        "category": "Geographic+Quality",
        "expected_fields": ["ocean", "avg_region", "PROFILE_TEMP_QC", "platform_number"]
    },
    
    # Complex multi-criteria queries
    {
        "id": 11,
        "query": "Which BCG floats in the Indian Ocean measure irradiance at different wavelengths?",
        "category": "Complex",
        "expected_fields": ["float_type", "ocean", "irr380_adjusted", "irr443_adjusted", "platform_number"]
    },
    
    {
        "id": 12,
        "query": "Find floats measuring cold water temperatures below 5 degrees in deep ocean",
        "category": "Complex",
        "expected_fields": ["temp_adjusted_c", "max_depth_m", "platform_number"]
    },
    
    {
        "id": 13,
        "query": "Show me all floats with nitrate measurements in tropical regions",
        "category": "Complex",
        "expected_fields": ["PROFILE_NITRATE_QC", "latitude", "platform_number"]
    },
    
    # Institution and profiler type queries
    {
        "id": 14,
        "query": "What APEX floats are available in the Atlantic Ocean?",
        "category": "FloatType",
        "expected_fields": ["platform_type", "ocean", "platform_number"]
    },
    
    # Temporal queries
    {
        "id": 15,
        "query": "Find floats with recent measurements from June 2023",
        "category": "Temporal",
        "expected_fields": ["measurement_date", "platform_number", "latitude"]
    },
    
    # Vague queries (testing semantic understanding)
    {
        "id": 16,
        "query": "Where is the warmest tropical water being measured?",
        "category": "Semantic",
        "expected_fields": ["temp_adjusted_c", "latitude", "longitude", "platform_number"]
    },
    
    {
        "id": 17,
        "query": "Show me data from the deepest measurements available",
        "category": "Semantic",
        "expected_fields": ["max_depth_m", "platform_number", "pres_adjusted_dbar"]
    },
    
    {
        "id": 18,
        "query": "Find ocean monitoring stations with the most comprehensive biogeochemical data",
        "category": "Semantic",
        "expected_fields": ["float_type", "cdom_adjusted", "PROFILE_CHLA_QC", "platform_number"]
    },
    
    # Cross-domain queries
    {
        "id": 19,
        "query": "Compare temperature and salinity profiles in polar vs tropical regions",
        "category": "CrossDomain",
        "expected_fields": ["latitude", "temp_adjusted_c", "psal_adjusted_psu", "platform_number"]
    },
    
    {
        "id": 20,
        "query": "Which floats have the most complete set of quality-checked measurements?",
        "category": "CrossDomain",
        "expected_fields": ["PROFILE_TEMP_QC", "PROFILE_PSAL_QC", "PROFILE_PRES_QC", "platform_number"]
    },
]


def print_test_queries():
    """Print all test queries in a readable format"""
    print("\n" + "=" * 80)
    print("RAG PIPELINE TEST QUERIES")
    print("=" * 80)
    
    # Group queries by category
    categories = {}
    for q in TEST_QUERIES:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q)
    
    # Print organized by category
    for category in sorted(categories.keys()):
        print(f"\n📚 {category.upper()}")
        print("-" * 80)
        
        for q in categories[category]:
            print(f"\n  [{q['id']:2d}] {q['query']}")
            print(f"       Expected Fields: {', '.join(q['expected_fields'])}")


def get_test_query(query_id: int) -> dict:
    """Get a specific test query by ID"""
    for q in TEST_QUERIES:
        if q["id"] == query_id:
            return q
    return None


def get_queries_by_category(category: str) -> list:
    """Get all queries in a specific category"""
    return [q for q in TEST_QUERIES if q["category"] == category]


def run_test_suite():
    """Run the entire test suite"""
    import subprocess
    import sys
    
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE RAG PIPELINE TEST SUITE")
    print("=" * 80)
    
    total_queries = len(TEST_QUERIES)
    passed = 0
    failed = 0
    
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{total_queries}] Testing Query ID {test['id']}: {test['category']}")
        print(f"    Query: {test['query']}")
        print("-" * 80)
        
        try:
            # Run RAG pipeline with this query
            result = subprocess.run(
                [sys.executable, "rag_pipeline.py", test["query"]],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print("    ✓ PASSED")
                passed += 1
            else:
                print("    ✗ FAILED")
                print(f"    Error: {result.stderr}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print("    ✗ TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            failed += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total_queries}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Success Rate: {(passed/total_queries)*100:.1f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--print":
            # Print all test queries
            print_test_queries()
        elif sys.argv[1] == "--run":
            # Run full test suite
            run_test_suite()
        elif sys.argv[1].isdigit():
            # Get specific test query
            query_id = int(sys.argv[1])
            q = get_test_query(query_id)
            if q:
                print(f"\nTest Query #{q['id']}:")
                print(f"Category: {q['category']}")
                print(f"Query: {q['query']}")
                print(f"Expected Fields: {', '.join(q['expected_fields'])}")
            else:
                print(f"Query ID {query_id} not found")
        else:
            # Get queries by category
            category = sys.argv[1]
            queries = get_queries_by_category(category)
            if queries:
                print(f"\nQueries in {category} category:")
                for q in queries:
                    print(f"  [{q['id']}] {q['query']}")
            else:
                print(f"Category '{category}' not found")
    else:
        # Default: print all queries
        print_test_queries()
