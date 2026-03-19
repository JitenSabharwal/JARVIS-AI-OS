from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import chromadb
import uuid

# ---------------------------
# LOAD MODEL
# ---------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# ---------------------------
# SETUP DATABASE
# ---------------------------
client = chromadb.Client()
collection = client.get_or_create_collection("jarvis_memory")

# ---------------------------
# ADD IMAGE TO MEMORY
# ---------------------------
def add_image(path, description=""):
    image = Image.open(path)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    image_features = model.get_image_features(**inputs)
    embedding = image_features[0].detach().numpy().tolist()

    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[description],
        metadatas=[{"type": "image", "path": path}]
    )

    print(f"✅ Stored: {path}")


# ---------------------------
# SEARCH MEMORY
# ---------------------------
def search(query):
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    text_features = model.get_text_features(**inputs)
    query_emb = text_features[0].detach().numpy().tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    return results


# ---------------------------
# TEST RUN
# ---------------------------
if __name__ == "__main__":
    
    # 👇 CHANGE THIS PATH
    add_image("/Users/sabharwal/Downloads/image.png", "a cat sitting on floor")

    results = search("cat")

    print("\n🔍 Results:")
    
    for i in range(len(results["ids"][0])):
        print("-----")
        print("Path:", results["metadatas"][0][i]["path"])
        print("Description:", results["documents"][0][i])