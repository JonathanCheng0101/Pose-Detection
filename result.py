from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

client = MongoClient(uri)
db     = client["pose_db_new"]
# Parameters
MONGO_URI = "mongodb+srv://JonCheng:Jona0101@cluster0.nhjixga.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME   = "pose_db_new"
COL_NAME  = "Jon"         # collection name for Jon’s shots
ANGLE_COLS = [
    'shoulder_angle','elbow_angle',
    'wrist_angle','hip_angle','knee_angle'
]
N_FRAMES  = n_frames      # e.g. 34

# 2) Load every player’s raw frame data into one DataFrame
all_docs = []
for col_name in db.list_collection_names():
    col_docs = list(db[col_name].find({}, {"_id":0}))
    for d in col_docs:
        d["player_name"] = col_name
    all_docs.extend(col_docs)
df = pd.DataFrame(all_docs)

# 3) Pick a fixed frame count (median of all shots)
frames_per_shot = (
    df.groupby(["player_name","video_id"])["frame"]
      .max()
      .reset_index(name="num_frames")
)
n_frames = int(round(frames_per_shot["num_frames"].median()))

# 4) Define the resampling helper
ANGLE_COLS = ["shoulder_angle","elbow_angle","wrist_angle","hip_angle","knee_angle"]
def resample_shot(shot_df):
    shot_df = shot_df.sort_values("frame").reset_index(drop=True)
    t0 = np.linspace(0,1,len(shot_df))
    t1 = np.linspace(0,1,n_frames)
    arrays = [interp1d(t0, shot_df[c], kind="linear", fill_value="extrapolate")(t1)
              for c in ANGLE_COLS]
    flat = np.hstack(arrays)
    return flat

# 5) Build one row per clip with its 5×n_frames vector
processed = []
for (player, vid), grp in df.groupby(["player_name","video_id"]):
    vec = resample_shot(grp)
    processed.append({"player_name":player, **{f"f{i}":v for i,v in enumerate(vec)}})
wide_df = pd.DataFrame(processed)

# 6) Compute each player’s average vector
feat_cols  = [c for c in wide_df.columns if c.startswith("f")]
wide_player = (
    wide_df
    .groupby("player_name")[feat_cols]
    .mean()
)


def resample_flat(shot_df):
    """
    Interpolate a single shot to N_FRAMES and flatten into
    a (5 * N_FRAMES,) vector of joint angles.
    """
    shot_df = shot_df.sort_values("frame").reset_index(drop=True)
    t_orig = np.linspace(0, 1, len(shot_df))
    t_new  = np.linspace(0, 1, N_FRAMES)
    arrays = [
        interp1d(t_orig, shot_df[col], kind='linear', fill_value='extrapolate')(t_new)
        for col in ANGLE_COLS
    ]
    return np.hstack(arrays)  # shape = (5 * N_FRAMES,)

# 1) Load Jon’s frames from MongoDB
client = MongoClient(MONGO_URI)
df_jon = pd.DataFrame(
    list(client[DB_NAME][COL_NAME].find({}, {"_id": 0}))
)
client.close()

# 2) Resample & flatten each video_id group, then average
if 'video_id' in df_jon.columns:
    vecs = [resample_flat(grp) for _, grp in df_jon.groupby('video_id')]
else:
    vecs = [resample_flat(df_jon)]
jon_avg = np.mean(vecs, axis=0).reshape(1, -1)  # shape = (1, 5*N_FRAMES)

# 3) Load precomputed wide_player DataFrame
#    index = player_name, columns = 5*N_FRAMES angle features
# wide_player = pd.read_pickle("wide_player.pkl")
players    = wide_player.index.to_list()
player_mat = wide_player.values               # shape = (n_players, 5*N_FRAMES)

# 4) Z-score normalize Jon and all players together
scaler      = StandardScaler().fit(np.vstack([jon_avg, player_mat]))
jon_norm    = scaler.transform(jon_avg)
players_norm = scaler.transform(player_mat)

# 5) Compute cosine similarities
sims = cosine_similarity(jon_norm, players_norm)[0]

# 6) Extract top-3 most similar (excluding Jon)
sorted_idx = np.argsort(-sims)
top3 = []
for i in sorted_idx:
    if players[i] == "Jon":
        continue
    top3.append((players[i], sims[i] * 100))
    if len(top3) == 3:
        break

print("Top-3 most similar players (normalized):")
for rank, (name, score) in enumerate(top3, start=1):
    print(f"{rank}. {name} — {score:.1f}%")
