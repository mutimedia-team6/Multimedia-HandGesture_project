import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

import inference
from heuristic import final_decision, assignment_raw_score


CLASS_NAMES = ["N/A", "fist", "like", "ok", "one", "palm"]
SEED = 42


def fix_path(path_str, base_dir):
    p = Path(path_str)
    return str(base_dir / p.parent.name / p.name)


def load_df():
    csv_path = Path("processed_hagrid_small/labels.csv")
    if not csv_path.exists():
        raise FileNotFoundError("找不到 processed_hagrid_small/labels.csv")

    df = pd.read_csv(csv_path)
    base_dir = csv_path.parent

    df["crop_path"] = df["crop_path"].apply(lambda x: fix_path(x, base_dir))
    df["landmark_path"] = df["landmark_path"].apply(lambda x: fix_path(x, base_dir))

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    return val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def evaluate_df(df, name="val"):
    y_true = []
    y_raw = []
    y_heur = []
    reasons = []

    for _, row in df.iterrows():
        img = np.array(Image.open(row["crop_path"]).convert("RGB"))
        lm = np.load(row["landmark_path"]).astype(np.float32)
        label = int(row["label"])

        logits = inference._model_forward(img, lm)

        raw_pred = int(np.argmax(logits))

        heur_pred, debug = final_decision(
            model_output=logits,
            landmarks=lm,
            assume_logits=True,
            return_debug=True,
        )

        y_true.append(label)
        y_raw.append(raw_pred)
        y_heur.append(int(heur_pred))
        reasons.append(debug["reason"])

    y_true = np.array(y_true)
    y_raw = np.array(y_raw)
    y_heur = np.array(y_heur)

    def metrics(pred):
        target_mask = y_true != 0
        na_mask = y_true == 0

        overall_acc = np.mean(pred == y_true)

        target_acc = (
            np.mean(pred[target_mask] == y_true[target_mask])
            if target_mask.sum() > 0 else 0.0
        )

        na_false_trigger_rate = (
            np.mean(pred[na_mask] != 0)
            if na_mask.sum() > 0 else 0.0
        )

        valid_reject_rate = (
            np.mean(pred[target_mask] == 0)
            if target_mask.sum() > 0 else 0.0
        )

        raw_score = assignment_raw_score(y_true, pred)

        return {
            "overall_acc": overall_acc,
            "target_acc": target_acc,
            "na_false_trigger_rate": na_false_trigger_rate,
            "valid_reject_rate": valid_reject_rate,
            "assignment_raw_score": raw_score,
        }

    raw_m = metrics(y_raw)
    heur_m = metrics(y_heur)

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
        pd.Series(y_raw, name="raw_pred"),
        dropna=False,
    ).reindex(index=range(6), columns=range(6), fill_value=0)

    cm_heur = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_heur, name="heur_pred"),
        dropna=False,
    ).reindex(index=range(6), columns=range(6), fill_value=0)

    print("\nRaw confusion matrix:")
    print(cm_raw)

    print("\nHeuristic confusion matrix:")
    print(cm_heur)


if __name__ == "__main__":
    val_df, test_df = load_df()

    # 先看 val，比較安全，避免一直對 test 調參
    evaluate_df(val_df, name="val")

    # 最後再看 test
    evaluate_df(test_df, name="test")