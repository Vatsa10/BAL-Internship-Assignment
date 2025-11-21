import asyncio
from main import AsyncContextRAG, COLLECTION_NAME

async def main():
    # Initialize RAG system
    rag = AsyncContextRAG()
    
    try:
        # Check if collection exists
        collection_info = await rag.client.get_collection(COLLECTION_NAME)
        count = await rag.client.count(collection_name=COLLECTION_NAME, exact=True)
        print(f"✅ Collection '{COLLECTION_NAME}' exists with {count.count} vectors")
        
        # Test queries
        test_queries = [
            "What is the GDP growth forecast?",
            "What are the key economic indicators?",
            "What are the main risks to the global economy?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"🔍 Query: {query}")
            try:
                answer, citations = await rag.query(query)
                print(f"\n💬 Answer: {answer}")
                if citations:
                    print(f"\n📚 Sources: {', '.join(citations)}")
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        print("\nYou may need to ingest documents first. Run the following command:")
        print("python main.py")
        print("Then use the web interface to upload your PDF document.")

if __name__ == "__main__":
    print("🚀 Testing RAG System")
    print("=" * 50)
    asyncio.run(main())
