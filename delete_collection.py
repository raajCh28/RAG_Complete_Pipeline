import os
import sys
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "localhost"),
    port=int(os.getenv("CHROMA_PORT", 8000)),
)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("COLLECTION_NAME")
    confirm = input(f"Type the collection name to confirm deleting '{name}': ").strip()
    if confirm == name:
        client.delete_collection(name)
        print(f"Deleted '{name}'.")
    else:
        print("Confirmation didn't match — nothing deleted.")