import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# ============================================================
# Same topology names used in the other multitopology scripts.
# ============================================================
TOPOLOGY_PATHS = {
    "us_ieee118": "data/processed/us_ieee118",
    "uk_ieee30": "data/processed/uk_ieee30",
    "fr_pegase89": "data/processed/fr_pegase89",
    "gb_full": "data/processed/gb_full",
    "eu_case300": "data/processed/eu_case300",
    "india_synthetic": "data/processed/india_synthetic",
}


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_features(scenario):
    weather = scenario["weather"]
    event = scenario["event"]
    total_wind = sum(v["wind_mw"] for v in scenario["renewable_output"].values())
    total_solar = sum(v["solar_mw"] for v in scenario["renewable_output"].values())
    features = [
        weather["wind_speed_kmh"],
        weather["solar_irradiance_wm2"],
        int(weather["storm_flag"]),
        scenario["load_multiplier"],
        total_wind,
        total_solar,
        1 if event["type"] == "line_outage" else 0,
        1 if event.get("cause") == "storm" else 0,
        1 if event.get("cause") == "equipment_failure" else 0,
    ]
    return features


def prepare_dataset(data):
    X = [extract_features(s) for s in data]
    y = [int(s["congested"]) for s in data]
    return np.array(X), np.array(y)


def train_baseline_for_topology(topology_name):
    base_dir = TOPOLOGY_PATHS[topology_name]
    train_path = f"{base_dir}/train.jsonl"
    val_path = f"{base_dir}/val.jsonl"
    test_path = f"{base_dir}/test.jsonl"
    model_out_path = f"app/model1_baseline_{topology_name}.pkl"
    results_out_path = f"experiments/model1_baseline_results_{topology_name}.txt"

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    test_data = load_jsonl(test_path)

    X_train, y_train = prepare_dataset(train_data)
    X_val, y_val = prepare_dataset(val_data)
    X_test, y_test = prepare_dataset(test_data)

    print(f"[{topology_name}] Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"[{topology_name}] Val:   {X_val.shape[0]} samples")
    print(f"[{topology_name}] Test:  {X_test.shape[0]} samples")

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    joblib.dump(model, model_out_path)
    print(f"[{topology_name}] Model 1 saved to {model_out_path}")

    val_preds = model.predict(X_val)
    print(f"\n[{topology_name}] --- Validation Set Performance ---")
    print(f"Accuracy:  {accuracy_score(y_val, val_preds):.3f}")
    print(f"Precision: {precision_score(y_val, val_preds):.3f}")
    print(f"Recall:    {recall_score(y_val, val_preds):.3f}")
    print(f"F1 Score:  {f1_score(y_val, val_preds):.3f}")

    test_preds = model.predict(X_test)
    print(f"\n[{topology_name}] --- Test Set Performance (FINAL) ---")
    print(f"Accuracy:  {accuracy_score(y_test, test_preds):.3f}")
    print(f"Precision: {precision_score(y_test, test_preds):.3f}")
    print(f"Recall:    {recall_score(y_test, test_preds):.3f}")
    print(f"F1 Score:  {f1_score(y_test, test_preds):.3f}")

    cm = confusion_matrix(y_test, test_preds)
    print(f"\n[{topology_name}] Confusion Matrix (Test Set):")
    print(f"                 Predicted: No   Predicted: Yes")
    print(f"Actual: No       {cm[0][0]:>13}   {cm[0][1]:>14}")
    print(f"Actual: Yes      {cm[1][0]:>13}   {cm[1][1]:>14}")

    feature_names = ["wind_speed", "solar_irradiance", "storm_flag", "load_multiplier",
                      "total_wind_mw", "total_solar_mw", "has_outage", "outage_storm", "outage_equipment"]
    print(f"\n[{topology_name}] Feature Importances:")
    for name, importance in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s}: {importance:.3f}")

    results_text = f"""
MODEL 1: STRUCTURED-ONLY BASELINE (Random Forest)
Topology: {topology_name}
Generated from simulation, {X_train.shape[0]+X_val.shape[0]+X_test.shape[0]} total scenarios

Dataset:
  Train: {X_train.shape[0]} samples
  Val:   {X_val.shape[0]} samples
  Test:  {X_test.shape[0]} samples

--- Validation Set Performance ---
Accuracy:  {accuracy_score(y_val, val_preds):.3f}
Precision: {precision_score(y_val, val_preds):.3f}
Recall:    {recall_score(y_val, val_preds):.3f}
F1 Score:  {f1_score(y_val, val_preds):.3f}

--- Test Set Performance (FINAL) ---
Accuracy:  {accuracy_score(y_test, test_preds):.3f}
Precision: {precision_score(y_test, test_preds):.3f}
Recall:    {recall_score(y_test, test_preds):.3f}
F1 Score:  {f1_score(y_test, test_preds):.3f}

Confusion Matrix (Test Set):
                 Predicted: No   Predicted: Yes
Actual: No       {cm[0][0]:>13}   {cm[0][1]:>14}
Actual: Yes      {cm[1][0]:>13}   {cm[1][1]:>14}

Feature Importances:
"""
    for name, importance in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        results_text += f"  {name:20s}: {importance:.3f}\n"

    with open(results_out_path, "w") as f:
        f.write(results_text)

    print(f"\n[{topology_name}] Results saved to {results_out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topology = sys.argv[1]
    else:
        topology = "uk_ieee30"
    train_baseline_for_topology(topology)