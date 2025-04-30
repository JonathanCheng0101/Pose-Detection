# main.py
import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_utils import load_data, preprocess_all
from train_mlp import train_model
from train_knn import loocv_knn

def run_preprocess(args):
    print("Loading raw data from MongoDB…")
    df = load_data(args.mongo_uri, args.db_name)
    print(f"Loaded {len(df)} rows.")

    print("Resampling each shot to fixed frame count…")
    final_df = preprocess_all(df)
    final_df.to_csv(args.output_frames, index=False)
    print(f"Preprocessed frames saved to {args.output_frames}")

    # also create clip-level table for KNN
    angle_cols = [c for c in final_df.columns if c.endswith("_angle")]
    wide = final_df.pivot(
        index=["player_name", "video_id"],
        columns="frame",
        values=angle_cols
    )
    wide.columns = [
        f"{ang}_frame{int(frm)}" for ang, frm in wide.columns
    ]
    wide.reset_index(inplace=True)
    wide.to_csv(args.output_clips, index=False)
    print(f"Clip-level features saved to {args.output_clips}")
    return final_df, wide

def run_mlp(args, final_df):
    print("Preparing MLP training data…")
    angle_cols = ["shoulder_angle","elbow_angle","wrist_angle","hip_angle","knee_angle"]
    le = LabelEncoder()
    final_df["label"] = le.fit_transform(final_df["player_name"])
    X = final_df[angle_cols].values
    y = final_df["label"].values

    # frame-level split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    print("Training Residual MLP…")
    model, scaler = train_model(
        X_tr, y_tr, X_val, y_val,
        num_classes=len(le.classes_),
        epochs=args.epochs
    )

    # save model and scaler
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch_path = os.path.join(args.checkpoint_dir, "resmlp2.pth")
    import torch
    torch.save(model.state_dict(), torch_path)
    print(f"Model weights saved to {torch_path}")

def run_knn(args, wide):
    print("Preparing KNN data…")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    wide["label"] = le.fit_transform(wide["player_name"])
    feat_cols = [c for c in wide.columns if c.startswith(tuple(["shoulder","elbow","wrist","hip","knee"]))]
    X = wide[feat_cols].values
    y = wide["label"].values

    print("Running leave-one-out KNN…")
    loocv_knn(X, y, k_list=[1,3,5,7])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline: preprocess → train MLP → evaluate KNN"
    )
    parser.add_argument("--mongo_uri", default=os.getenv("MONGO_URI"),
                        help="MongoDB connection URI")
    parser.add_argument("--db_name", default=os.getenv("DB_NAME","pose_db_new"),
                        help="MongoDB database name")
    parser.add_argument("--output_frames", default="preprocessed_frames.csv",
                        help="where to save frame-level CSV")
    parser.add_argument("--output_clips", default="clip_features.csv",
                        help="where to save clip-level CSV")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="where to save trained model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of MLP training epochs")

    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("preprocess", help="run data loading & preprocessing")
    sub.add_parser("train-mlp", help="train the MLP model")
    sub.add_parser("train-knn", help="evaluate KNN baseline")

    args = parser.parse_args()
    if args.cmd == "preprocess":
        final_df, wide = run_preprocess(args)
    elif args.cmd == "train-mlp":
        # need final_df in memory
        df = pd.read_csv(args.output_frames)
        final_df, _ = None, None
        # recover labels & frames
        final_df = df
        run_mlp(args, final_df)
    elif args.cmd == "train-knn":
        wide = pd.read_csv(args.output_clips)
        run_knn(args, wide)
    else:
        parser.print_help()
