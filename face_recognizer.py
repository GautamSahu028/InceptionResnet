from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os
from torchvision import transforms
from torch.nn.functional import cosine_similarity


class FaceRecognizer:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.known_embeddings = {}
        self.unknown_embeddings = {}

    def get_embedding(self, img):
        face = self.mtcnn(img)
        if face is None:
            return None
        if isinstance(face, list) or face.ndim == 4:
            face = face[0]  # take only the first face if multiple
        face = face.unsqueeze(0).to(self.device)
        return self.model(face)
    
    def train_on_multiple(self, training_dir):
        for person_name in os.listdir(training_dir):
            person_path = os.path.join(training_dir, person_name)

            if not os.path.isdir(person_path):
                continue

            embeddings = []

            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)

                if not os.path.isfile(img_path):
                    continue

                try:
                    img = Image.open(img_path)
                    emb = self.get_embedding(img)
                    if emb is not None:
                        embeddings.append(emb)
                    else:
                        print(f"Face not detected in {img_file}")
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")

            if embeddings:
                avg_embedding = torch.stack(embeddings).mean(dim=0)
                self.known_embeddings[person_name] = avg_embedding


    def train(self, training_dir):
        for filename in os.listdir(training_dir):
            full_path = os.path.join(training_dir, filename)

            if not os.path.isfile(full_path) or filename.lower() == 'test.jpg':
                continue

            name = os.path.splitext(filename)[0]

            try:
                img = Image.open(full_path)
                emb = self.get_embedding(img)

                if emb is not None:
                    self.known_embeddings[name] = emb
                else:
                    print(f"Face not detected in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    def detect_faces(self, img_path):
        if isinstance(img_path, str):  # if path is given
            img = Image.open(img_path)
        else:  # assume it's already a PIL.Image
            img = img_path
        faces = self.mtcnn(img)

        if faces is not None:
            for i, face in enumerate(faces):
                face = (face + 1) / 2
                face_img = transforms.ToPILImage()(face.cpu().detach())
                emb = self.get_embedding(face_img)
                if emb is not None:
                    self.unknown_embeddings[f'face_{i+1}'] = emb
        else:
            print("No faces detected.")

    def predict_name(self, test_emb, threshold=0.6):
        best_match = None
        best_score = -1

        for name, emb in self.known_embeddings.items():
            sim = cosine_similarity(test_emb, emb).item()
            if sim > best_score:
                best_score = sim
                best_match = name

        return (best_match, best_score) if best_score >= threshold else ("Unknown", best_score)

    def predict_unknown_faces(self):
        for key, test_emb in self.unknown_embeddings.items():
            predicted_name, similarity_score = self.predict_name(test_emb)
            print(f"{key}: Predicted - {predicted_name} (Similarity: {similarity_score:.3f})")
    
    def save_embeddings(self, path='embeddings.pth'):
        torch.save(self.known_embeddings, path)
        print(f"Embeddings saved to {path}")

    def load_embeddings(self, path='embeddings.pth'):
        if os.path.exists(path):
            self.known_embeddings = torch.load(path, map_location=self.device)
            print(f"Embeddings loaded from {path}")
        else:
            print(f"No saved embeddings found at {path}")


# if __name__ == "__main__":
#     recognizer = FaceRecognizer()

#     if os.path.exists("embeddings.pth"):
#         recognizer.load_embeddings("embeddings.pth")
#     else:
#         recognizer.train_on_multiple("static/NITR")
#         # recognizer.train_on_multiple("static/GOT2")
#         recognizer.save_embeddings("embeddings.pth")

#     recognizer.detect_faces("train.jpg")
#     recognizer.predict_unknown_faces()
