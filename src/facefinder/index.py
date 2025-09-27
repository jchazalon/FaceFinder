import os
import json
from datetime import datetime
# import hashlib
from jinja2 import Environment, FileSystemLoader
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
import faiss
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_thumbnail(image, region, out_path, size=(160, 160)):
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    face = image.crop((x, y, x + w, y + h)).resize(size)
    face.save(out_path)


# ----------------------------
# Face Detection
# ----------------------------

def detect_faces(image, detector):
    image_array = np.asarray(image)
    detections = detector.detect_faces(image_array)

    faces = []
    for det in detections:
        box = det["box"]  # [x, y, width, height]
        faces.append({
            "x": int(box[0]),
            "y": int(box[1]),
            "width": int(box[2]),
            "height": int(box[3])
        })
    return faces


# ----------------------------
# Face Embedding
# ----------------------------

def embed_faces(image, regions, model, device):
    embeddings = []

    for region in regions:
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        face = image.crop((x, y, x + w, y + h))

        face = face.resize((160, 160))
        face_tensor = torch.tensor(np.array(face)).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(face_tensor).cpu().numpy().flatten().tolist()

        embeddings.append(embedding)

    return embeddings


# ----------------------------
# HTML Preview Generator
# ----------------------------

def generate_html_preview(metadata, result_indices, thumbnails_dir, output_html, page_size=50):
    """
    Generates an HTML file for face search results using an external template.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_html) or '.', exist_ok=True)

    # Prepare items
    items = [metadata[i] for i in result_indices]
    for it in items:
        it['thumbnail'] = os.path.join(thumbnails_dir, os.path.basename(it['thumbnail'])).replace('\\', '/')
        it['full_path'] = it['file'].replace('\\', '/')

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader(searchpath="templates"), autoescape=True)
    template = env.get_template(os.path.basename("results.html"))

    # Render template
    rendered_html = template.render(items_json=json.dumps(items), page_size=page_size)

    # Write to output
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(rendered_html)

    print(f"✅ HTML preview written to {output_html}")


# ----------------------------
# Indexing
# ----------------------------

def build_index(input_folder, output_folder):
    index_file = os.path.join(output_folder, "faces.index")
    embedding_file = os.path.join(output_folder, "embeddings.npy")
    metadata_file = os.path.join(output_folder, "metadata.json")
    thumbnails_dir = os.path.join(output_folder, "thumbnails")
    log_file = os.path.join(output_folder, "indexing.log")
    ensure_dir(thumbnails_dir)

    input_folder = os.path.abspath(input_folder)

    detector = MTCNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # ensure_dir(output_folder)

    embeddings = []
    # Load existing embedding_file and metadata if they exist
    if os.path.exists(embedding_file):
        embeddings = np.load(embedding_file).tolist()
        print(f"Loaded {len(embeddings)} existing embeddings.")
    else:
        embeddings = []


    # Reopen the medatada file to check for existing IDs
    metadata = []
    if os.path.exists(metadata_file):
        with open(metadata_file, encoding="utf8") as f:
            try:
                metadata = json.load(f)
            except Exception:
                metadata = []
    
    # Check embeddings and metadata consistency
    print(f"Loaded {len(metadata)} existing metadata entries.")
    if len(embeddings) != len(metadata):
        print("⚠️ Warning: Embeddings and metadata count mismatch. Rebuilding index from scratch.")
        embeddings = []
        metadata = []

    # Recursively find files in input_folder and collect absolute paths
    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))

    # Filter files which already exist in metadata
    existing_files = [item['file'] for item in metadata]
    image_files = [f for f in image_files if f.replace('\\', '/') not in existing_files]
    print(f"Found {len(image_files)} new images to process.")
    id_counter = len(metadata)

    existing_files = [item['file'] for item in metadata]
    print(f"Already indexed {len(existing_files)} images, skipping them.")
    
    # Open log file and write the timestamp
    with open(log_file, "a", encoding="utf8", buffering=1) as logf:

        logf.write("="*50 + "\n")
        logf.write(f"\nIndexing run at {datetime.now().isoformat()}\n")
        logf.write(f"Input folder: {input_folder}\n")
        logf.write(f"Output folder: {output_folder}\n")
        logf.write(f"Existing indexed files: {len(existing_files)}\n")
        logf.write(f"New files to process: {len(image_files)}\n")
        # List files already indexed
        logf.write("Already indexed files:\n")
        for ef in existing_files:
            logf.write(f"  - {ef}\n")
        logf.write("\n")
        logf.write("Starting indexing of new files.\n")


        # id_counter = 0
        for image_path in tqdm(image_files, desc="Indexing images"):
            logf.write(f"Processing {image_path}\n")
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Could not open image {image_path}: {e}")
                logf.write(f"  [ERROR] Could not open image: {e}\n")
                continue

            # # Generate a unique ID based on the image path
            # id_hash = hashlib.md5(image_path.encode('utf-8')).hexdigest()
            # # Append something depending on the filename to avoid collisions
            # clean_basename = os.path.basename(image_path).replace(' ', '_').replace('.', '_').replace('-', '_')[:10]
            # id_hash += f"_{clean_basename}"

            try:
                faces = detect_faces(image, detector)
                logf.write(f"  Detected {len(faces)} faces.\n")
                if not faces:
                    continue

                face_embeddings = embed_faces(image, faces, model, device)

                for region, emb in zip(faces, face_embeddings):
                    tid = f"{id_counter:08d}.jpg"
                    thumb_path = os.path.join(thumbnails_dir, tid)
                    save_thumbnail(image, region, thumb_path)

                    embeddings.append(emb)
                    metadata.append({
                        "id": id_counter,
                        "file": image_path.replace('\\', '/'),
                        "region": region,
                        "thumbnail": thumb_path.replace('\\', '/')
                    })
                    id_counter += 1
            except Exception as e:
                print(f"[WARN] Could not process {image_path}: {e}")
                logf.write(f"  [ERROR] Could not process image: {e}\n")
            
            # Periodically save progress
            if len(embeddings) % 50 == 0:
                np.save(embedding_file, np.array(embeddings).astype("float32"))
                with open(metadata_file, "w", encoding="utf8") as f:
                    json.dump(metadata, f, indent=2)
                print(f"Saved progress: {len(embeddings)} embeddings.")
                logf.write(f"  Saved {len(embeddings)} embeddings.\n")

        if not embeddings:
            print("⚠️ No faces found in folder.")
            return

        embeddings = np.array(embeddings).astype("float32")
        np.save(embedding_file, embeddings)


        logf.write(f"Finished indexing. Total embeddings: {len(embeddings)}\n")
        logf.write(f"Regenerating FAISS index at {index_file}\n")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        # pylint: disable=E1120
        index.add(embeddings)

        faiss.write_index(index, index_file)
        with open(metadata_file, "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Indexed {len(embeddings)} faces. Index saved to {index_file}, metadata to {metadata_file}.")
        logf.write(f"FAISS index saved to {index_file}\n")
        logf.write(f"Metadata saved to {metadata_file}\n")

    # Also generate a full HTML preview (paged)
    # try:
    #     generate_html_preview(metadata, list(range(len(metadata))), output_html='index_preview.html', thumbnails_dir=thumbnails_dir)
    # except Exception as e:
    #     print(f"[WARN] Could not generate HTML preview: {e}")


# ----------------------------
# Querying
# ----------------------------

def query(sample_image_path, index_folder, output_folder, topk=50):
    index_file = os.path.join(index_folder, "faces.index")
    metadata_file = os.path.join(index_folder, "metadata.json")
    thumbnails_dir = os.path.join(index_folder, "thumbnails")
    
    output_html = os.path.join(output_folder, 'query_preview.html')
    ensure_dir(output_folder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = MTCNN()
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Load FAISS index and metadata
    index = faiss.read_index(index_file)
    with open(metadata_file, encoding="utf8") as f:
        metadata = json.load(f)

    # Open the image
    sample_image = Image.open(sample_image_path).convert("RGB")

    # Detect + embed face from query image
    faces = detect_faces(sample_image, detector)
    if not faces:
        print("⚠️ No face detected in sample image.")
        return

    query_embeddings = embed_faces(sample_image, [faces[0]], model, device)
    query_embedding = np.array(query_embeddings).astype("float32")

    D, I = index.search(query_embedding, k=topk)

    print("🔍 Top matches:")
    result_indices = []
    for idx, dist in zip(I[0], D[0]):
        print(metadata[idx], "distance:", float(dist))
        result_indices.append(idx)

    # generate HTML preview for the returned items
    try:
        generate_html_preview(metadata, result_indices, thumbnails_dir, output_html)
        print(f"Preview HTML saved to {output_html}")
    except Exception as e:
        print(f"[WARN] Could not generate query HTML preview: {e}")


