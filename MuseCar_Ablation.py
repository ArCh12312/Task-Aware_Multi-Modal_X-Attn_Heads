from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
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

val_dataset = MultimodalDataset(
    dev_labels,
    video_dir,
    audio_dir,
    text_dir,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    pin_memory=True
)

config = BertConfig.from_pretrained("Task_Aware_Multi_Modal_X_Attn_Heads/my_custom_model/")
model = CustomBertMTL(config)
model.load_state_dict(torch.load("Muse_Uniform_model.bin", weights_only=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def cache_pooled(model, val_loader, device):

    pooled_list = []
    arousal_labels = []
    valence_labels = []
    topic_labels = []

    with torch.inference_mode():

        for batch in tqdm(val_loader, desc="Caching pooled"):

            text  = batch["text"].to(device)
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)

            _ = model(video, audio, text)

            pooled = model.last_pooled.detach().cpu()

            pooled_list.append(pooled)

            arousal_labels.append(batch["arousal"])
            valence_labels.append(batch["valence"])
            topic_labels.append(batch["topic"])

    pooled_all = torch.cat(pooled_list)
    arousal_labels = torch.cat(arousal_labels)
    valence_labels = torch.cat(valence_labels)
    topic_labels = torch.cat(topic_labels)

    return pooled_all, arousal_labels, valence_labels, topic_labels
  
def evaluate_cached(
    pooled_all,
    arousal_labels,
    valence_labels,
    topic_labels,
    model,
    mode
):

    pooled = pooled_all.clone().to(device)

    # -------- REMOVE BLOCKS --------
    if mode == "remove_y1":
        pooled[:, 0:256] = 0

    elif mode == "remove_y2":
        pooled[:, 256:512] = 0

    elif mode == "remove_y3":
        pooled[:, 512:768] = 0

    # -------- KEEP ONLY BLOCKS --------
    elif mode == "only_y1":
        pooled[:, 256:768] = 0

    elif mode == "only_y2":
        pooled[:, 0:256] = 0
        pooled[:, 512:768] = 0

    elif mode == "only_y3":
        pooled[:, 0:512] = 0

    with torch.inference_mode():

        arousal_logits = model.arousal_head(pooled)
        valence_logits = model.valence_head(pooled)
        topic_logits   = model.topic_head(pooled)

    arousal_pred = torch.argmax(arousal_logits, dim=1).cpu().numpy()
    valence_pred = torch.argmax(valence_logits, dim=1).cpu().numpy()
    topic_pred   = torch.argmax(topic_logits, dim=1).cpu().numpy()

    f1_arousal = f1_score(arousal_labels.numpy(), arousal_pred, average="macro")
    f1_valence = f1_score(valence_labels.numpy(), valence_pred, average="macro")
    f1_topic   = f1_score(topic_labels.numpy(), topic_pred, average="macro")

    return f1_arousal, f1_valence, f1_topic

pooled_all, arousal_labels, valence_labels, topic_labels = cache_pooled(
    model,
    val_loader,
    device
)  
    
modes = [
    "baseline",
    "remove_y1",
    "remove_y2",
    "remove_y3",
    "only_y1",
    "only_y2",
    "only_y3"
]

for mode in modes:

    f1_a, f1_v, f1_t = evaluate_cached(
        pooled_all,
        arousal_labels,
        valence_labels,
        topic_labels,
        model,
        mode
    )

    print(f"\nMode: {mode}")
    print(f"Arousal F1: {f1_a:.4f}")
    print(f"Valence F1: {f1_v:.4f}")
    print(f"Topic F1:   {f1_t:.4f}")