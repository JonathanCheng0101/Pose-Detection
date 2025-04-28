import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---- 0. Load .env and read Mongo URI ----
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# ---- Configuration ----
DB_NAME   = "pose_db_new"
COL_NAME  = "Jon"   # collection name for Jon’s shots
ANGLE_COLS = [
    "shoulder_angle", "elbow_angle",
    "wrist_angle", "hip_angle", "knee_angle"
]

# ---- 1. Connect & load all players’ frame data ----
client = MongoClient(MONGO_URI)
db     = client[DB_NAME]

all_docs = []
for col in db.list_collection_names():
    docs = list(db[col].find({}, {"_id": 0}))
    for d in docs:
        d["player_name"] = col
    all_docs.extend(docs)
df = pd.DataFrame(all_docs)

# ---- 2. Determine fixed frame count (median length) ----
frames_per_shot = (
    df.groupby(["player_name", "video_id"])["frame"]
      .max()
      .reset_index(name="num_frames")
)
n_frames = int(round(frames_per_shot["num_frames"].median()))

# ---- 3. Resampling helper ----
def resample_flat(shot_df):
    shot_df = shot_df.sort_values("frame").reset_index(drop=True)
    t0 = np.linspace(0, 1, len(shot_df))
    t1 = np.linspace(0, 1, n_frames)
    arrays = [
        interp1d(t0, shot_df[col], kind="linear", fill_value="extrapolate")(t1)
        for col in ANGLE_COLS
    ]
    return np.hstack(arrays)

# ---- 4. Build wide_df for all players ----
processed = []
for (player, vid), grp in df.groupby(["player_name", "video_id"]):
    vec = resample_flat(grp)
    processed.append({
        "player_name": player,
        **{f"f{i}": v for i, v in enumerate(vec)}
    })
wide_df = pd.DataFrame(processed)

# ---- 5. Compute each player’s average vector ----
feat_cols   = [c for c in wide_df.columns if c.startswith("f")]
wide_player = wide_df.groupby("player_name")[feat_cols].mean()

# ---- 6. Load Jon’s shots & compute his average vector ----
jon_df = pd.DataFrame(list(db[COL_NAME].find({}, {"_id": 0})))
if "video_id" in jon_df:
    jon_vecs = [resample_flat(grp) for _, grp in jon_df.groupby("video_id")]
else:
    jon_vecs = [resample_flat(jon_df)]
jon_avg = np.mean(jon_vecs, axis=0).reshape(1, -1)

client.close()

# ---- 7. Normalize & compute cosine similarities ----
player_mat   = wide_player.values
scaler       = StandardScaler().fit(np.vstack([jon_avg, player_mat]))
jon_norm     = scaler.transform(jon_avg)
players_norm = scaler.transform(player_mat)
sims         = cosine_similarity(jon_norm, players_norm)[0]

# ---- 8. Report top-3 most similar players (excluding Jon) ----
players = wide_player.index.to_list()
top3 = [
    (players[i], sims[i] * 100)
    for i in sims.argsort()[::-1]
    if players[i] != COL_NAME
][:3]

print("Top-3 most similar players:")
for rank, (name, score) in enumerate(top3, 1):
    print(f"{rank}. {name} — {score:.1f}%")
