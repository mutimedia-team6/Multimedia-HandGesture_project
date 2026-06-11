import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import inference
from inference import predict, final_decision, assignment_raw_score

SEED = 42


def restore_landmark_pixel_coords(cropped_img, landmarks):
    lm = np.asarray(landmarks, dtype=np.float32).copy()

    if cropped_img is not None and cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
        h, w = cropped_img.shape[:2]
        lm[:, 0] *= w
        lm[:, 1] *= h

    return lm


def load_df():
    csv_path = Path("dataset_v3_processed_detected/labels_fixed.csv")
    if not csv_path.exists():
        raise FileNotFoundError("找不到 processed_hagrid_small/labels_fixed.csv")

    df = pd.read_csv(csv_path)
    base_dir = csv_path.parent

    def fix_path(p):
        p = Path(str(p).replace("\\", "/"))
        return str(base_dir / p.parent.name / p.name)

    df["crop_path"] = df["crop_path"].apply(fix_path)
    df["landmark_path"] = df["landmark_path"].apply(fix_path)

    _, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"])

    return val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_metrics(y_true, y_pred):
    target_mask = y_true != 0
    na_mask = y_true == 0
    return {
        "overall_acc":            np.mean(y_pred == y_true),
        "target_acc":             np.mean(y_pred[target_mask] == y_true[target_mask]),
        "na_false_trigger_rate":  np.mean(y_pred[na_mask] != 0),
        "valid_reject_rate":      np.mean(y_pred[target_mask] == 0),
        "assignment_raw_score":   assignment_raw_score(y_true, y_pred),
    }


def evaluate_df(df, name="val"):
    y_true, y_raw, y_heur, reasons = [], [], [], []

    for _, row in df.iterrows():
        img  = np.array(Image.open(row["crop_path"]).convert("RGB"))
        lm   = np.load(row["landmark_path"]).astype(np.float32)
        lm_pixel = restore_landmark_pixel_coords(img, lm)
        true = int(row["label"])

        logits = inference._model_forward(img, lm)

        raw_pred = int(np.argmax(logits))

        heur_pred, debug = final_decision(
            model_output=logits,
            landmarks=lm_pixel,
            assume_logits=True,
            return_debug=True,
        )

        y_true.append(true)
        y_raw.append(raw_pred)
        y_heur.append(int(heur_pred))
        reasons.append(debug["reason"])

    y_true = np.array(y_true)
    y_raw  = np.array(y_raw)
    y_heur = np.array(y_heur)

    raw_m  = compute_metrics(y_true, y_raw)
    heur_m = compute_metrics(y_true, y_heur)

    print(f"\n===== {name.upper()} SET =====")
    print(f"num samples: {len(df)}")

    print("\n[Raw model]")
    for k, v in raw_m.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n[With heuristic]")
    for k, v in heur_m.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n[Delta: heuristic - raw]")
    for k in raw_m:
        diff = heur_m[k] - raw_m[k]
        print(f"{k}: {diff:+.4f}" if isinstance(diff, float) else f"{k}: {diff:+d}")

    print("\nHeuristic rejection reasons:")
    print(pd.Series(reasons).value_counts())

    cm_raw = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_raw,  name="raw_pred"),
        dropna=False,
    ).reindex(index=range(6), columns=range(6), fill_value=0)

    cm_heur = pd.crosstab(
        pd.Series(y_true,  name="true"),
        pd.Series(y_heur, name="heur_pred"),
        dropna=False,
    ).reindex(index=range(6), columns=range(6), fill_value=0)

    print("\nRaw confusion matrix:")
    print(cm_raw)

    print("\nHeuristic confusion matrix:")
    print(cm_heur)


if __name__ == "__main__":
    val_df, test_df = load_df()
    evaluate_df(val_df,  name="val")
    evaluate_df(test_df, name="test")