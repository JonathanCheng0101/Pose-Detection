from pymongo import MongoClient
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def load_data(uri: str, db_name: str) -> pd.DataFrame:
    """
    Connect to MongoDB and load all player shot documents into one DataFrame.
    """
    client = MongoClient(uri)
    db = client[db_name]
    records = []
    for col in db.list_collection_names():
        docs = list(db[col].find({}, projection={"_id": 0}))
        for d in docs:
            d["player_name"] = col
        records.extend(docs)
    return pd.DataFrame(records)

def resample_shot(shot_df: pd.DataFrame, n_frames: int,
                  angle_cols=None, method='linear') -> pd.DataFrame:
    """
    Interpolate a single shot to `n_frames` via linear (or other) interpolation.
    """
    shot_df = shot_df.sort_values('frame').reset_index(drop=True)
    orig_len = len(shot_df)
    t_orig = np.linspace(0, 1, orig_len)
    t_new  = np.linspace(0, 1, n_frames)
    if angle_cols is None:
        angle_cols = [c for c in shot_df.columns if c.endswith('_angle')]

    new_data = {'frame': np.arange(1, n_frames+1)}
    for col in angle_cols:
        f = interp1d(t_orig, shot_df[col], kind=method, fill_value='extrapolate')
        new_data[col] = f(t_new)

    # preserve identifiers
    new_data['player_name'] = [shot_df['player_name'].iloc[0]] * n_frames
    new_data['video_id']    = [shot_df['video_id'].iloc[0]]   * n_frames
    return pd.DataFrame(new_data)

def preprocess_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample every (player, video) clip to the median frame count,
    then concatenate into one DataFrame with a `shot_id` field.
    """
    # compute median number of frames per clip
    max_frames = df.groupby(['player_name','video_id'])['frame'] \
                   .max().reset_index(name='num_frames')
    med = int(round(max_frames['num_frames'].median()))

    processed = []
    for (_, _), grp in df.groupby(['player_name','video_id']):
        processed.append(resample_shot(grp, med))
    final_df = pd.concat(processed, ignore_index=True)
    final_df['shot_id'] = final_df['player_name'] + "__" + final_df['video_id'].astype(str)
    return final_df
