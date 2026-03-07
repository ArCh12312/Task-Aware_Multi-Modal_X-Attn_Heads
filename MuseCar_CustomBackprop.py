from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
from transformers import BertConfig
from Task_Aware_Multi_Modal_X_Attn_Heads.customBERT import CustomBertMTL
from Task_Aware_Multi_Modal_X_Attn_Heads.MuseCar_Embedding_generation.utils import load_embeddings, load_labels

video_dir = "Task_Aware_Multi_Modal_X_Attn_Heads/MuseCar_Sent/Video_output_512_DINOv2"
audio_dir = "Task_Aware_Multi_Modal_X_Attn_Heads/MuseCar_Sent/Audio_output_512_HuBERT"
text_dir = "Task_Aware_Multi_Modal_X_Attn_Heads/MuseCar_Sent/Text_output_512_BERT"
train_labels_file = "Task_Aware_Multi_Modal_X_Attn_Heads/MuseCar_Sent/train.csv"
dev_labels_file = "Task_Aware_Multi_Modal_X_Attn_Heads/MuseCar_Sent/devel.csv"

train_labels = load_labels(train_labels_file)
dev_labels = load_labels(dev_labels_file)

print("Loaded labels")

class MultimodalDataset(Dataset):
    def __init__(self, labels, video_dir, audio_dir, text_dir):
        self.labels = labels

        self.video_dir = Path(video_dir)
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)

        self.keys = []
        self.loaded_count = 0

        for video_id, scene_id in labels.keys():

            vid = self.video_dir / f"{video_id}_scene_{scene_id}.npy"
            audio = self.audio_dir / f"{video_id}_{scene_id}.npy"
            text = self.text_dir / f"{video_id}_scene{scene_id}.npy"

            if vid.exists() and audio.exists() and text.exists():
                self.keys.append((video_id, scene_id))
            #else:
            #    print(f"Skipping {(video_id, scene_id)} - missing embedding")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        video_id, scene_id = self.keys[idx]

        vid = self.video_dir / f"{video_id}_scene_{scene_id}.npy"
        audio = self.audio_dir / f"{video_id}_{scene_id}.npy"
        text = self.text_dir / f"{video_id}_scene{scene_id}.npy"

        video_emb = torch.from_numpy(np.load(vid)).float()
        audio_emb = torch.from_numpy(np.load(audio)).float()
        text_emb = torch.from_numpy(np.load(text)).float()

        label = self.labels[(video_id, scene_id)]

        return {
            "video": video_emb,
            "audio": audio_emb,
            "text": text_emb,
            "arousal": torch.tensor(label["arousal"]),
            "valence": torch.tensor(label["valence"]),
            "topic": torch.tensor(label["topic"])
        }
        
train_dataset = MultimodalDataset(
    train_labels,
    video_dir,
    audio_dir,
    text_dir,
)

val_dataset = MultimodalDataset(
    dev_labels,
    video_dir,
    audio_dir,
    text_dir,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    pin_memory=True
)

print("Loaded Datasets + Dataloaders")

config = BertConfig.from_pretrained("Task_Aware_Multi_Modal_X_Attn_Heads/my_custom_model/")
model = CustomBertMTL(config)
model.load_state_dict(torch.load("Muse_Uniform_model.bin", weights_only=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Loaded Model")
print("Device: ", device)

task_weights = torch.tensor([
    [1.00, 0.00, 0.00],  # arousal
    [0.00, 0.50, 0.50],  # valence
    [0.00, 0.20, 0.80],  # topic
], device=device)

video_params = list(model.subspace_proj.proj1.parameters())
audio_params = list(model.subspace_proj.proj2.parameters())
text_params  = list(model.subspace_proj.proj3.parameters())

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    video_params,
    audio_params,
    text_params,
    epochs=5,
    scheduler=None,
):

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        # -----------------------
        # Training
        # -----------------------
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):

            text  = batch["text"].to(device)
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)

            arousal = batch["arousal"].to(device)
            valence = batch["valence"].to(device)
            topic   = batch["topic"].to(device)

            optimizer.zero_grad()

            arousal_logits, valence_logits, topic_logits = model(video, audio, text)
            
            loss_arousal = criterion(arousal_logits, arousal)
            loss_valence = criterion(valence_logits, valence)
            loss_topic   = criterion(topic_logits, topic)
            
            total_loss = loss_arousal + loss_valence + loss_topic
            
            task_losses = torch.stack([
                    loss_arousal,
                    loss_valence,
                    loss_topic
                ])
            modality_losses = task_weights.T @ task_losses

            modality_losses[0].backward(inputs=video_params, retain_graph=True)
            modality_losses[1].backward(inputs=audio_params, retain_graph=True)
            modality_losses[2].backward(inputs=text_params)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if scheduler:
                scheduler.step()

            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        val_loss = 0.0

        arousal_preds, valence_preds, topic_preds = [], [], []
        arousal_labels, valence_labels, topic_labels = [], [], []

        with torch.no_grad():
            for batch in val_loader:

                text  = batch["text"].to(device)
                video = batch["video"].to(device)
                audio = batch["audio"].to(device)

                arousal = batch["arousal"].to(device)
                valence = batch["valence"].to(device)
                topic   = batch["topic"].to(device)

                arousal_logits, valence_logits, topic_logits = model(video, audio, text)

                loss_arousal = criterion(arousal_logits, arousal)
                loss_valence = criterion(valence_logits, valence)
                loss_topic   = criterion(topic_logits, topic)

                loss = loss_arousal + loss_valence + loss_topic
                val_loss += loss.item()

                arousal_pred = torch.argmax(arousal_logits, dim=1)
                valence_pred = torch.argmax(valence_logits, dim=1)
                topic_pred   = torch.argmax(topic_logits, dim=1)

                arousal_preds.append(arousal_pred.cpu())
                valence_preds.append(valence_pred.cpu())
                topic_preds.append(topic_pred.cpu())

                arousal_labels.append(arousal.cpu())
                valence_labels.append(valence.cpu())
                topic_labels.append(topic.cpu())

        arousal_preds = torch.cat(arousal_preds).numpy()
        valence_preds = torch.cat(valence_preds).numpy()
        topic_preds   = torch.cat(topic_preds).numpy()

        arousal_labels = torch.cat(arousal_labels).numpy()
        valence_labels = torch.cat(valence_labels).numpy()
        topic_labels   = torch.cat(topic_labels).numpy()

        f1_arousal = f1_score(arousal_labels, arousal_preds, average="macro")
        f1_valence = f1_score(valence_labels, valence_preds, average="macro")
        f1_topic   = f1_score(topic_labels, topic_preds, average="macro")

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Arousal F1: {f1_arousal:.4f}")
        print(f"Valence F1: {f1_valence:.4f}")
        print(f"Topic F1:   {f1_topic:.4f}")
        print("-" * 40)

    return model

for param in model.parameters():
    param.requires_grad = False

for param in model.arousal_head.parameters():
    param.requires_grad = True

for param in model.valence_head.parameters():
    param.requires_grad = True

for param in model.topic_head.parameters():
    param.requires_grad = True

for param in model.subspace_proj.parameters():
    param.requires_grad = True
    
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5
)

trained_model = train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    video_params,
    audio_params,
    text_params,
    epochs=10
)

# torch.save(trained_model.state_dict(), "Muse_Uniform_model.bin")

