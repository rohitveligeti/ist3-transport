import pandas as pd

IST3_PATH = "df.sas7bdat"
IST1_PATH = "ist1.CSV"
OUTPUT_PATH = "ist_combined.csv"

SHARED = ["source", "age", "delay_hours", "sbp", "male",
          "atrial_fib", "visible_infarct", "treated", "alive_independent"]


def load_ist3(path):
    raw = pd.read_sas(path)
    df = pd.DataFrame()
    df["age"]             = pd.to_numeric(raw["age"], errors="coerce")
    df["delay_hours"]     = pd.to_numeric(raw["randdelay"], errors="coerce")
    df["sbp"]             = pd.to_numeric(raw["sbprand"], errors="coerce")
    df["male"]            = pd.to_numeric(raw["gender"], errors="coerce").map({2: 1, 1: 0})        # 1=F,2=M
    df["atrial_fib"]      = pd.to_numeric(raw["atrialfib_rand"], errors="coerce").map({1: 1, 2: 0})  # 1=Yes,2=No
    df["visible_infarct"] = pd.to_numeric(raw["vis_infarct"], errors="coerce").map({1: 1, 2: 0})     # 1=Yes,2=No
    # outcome: alive & independent at 6 months (aliveind6: 1=Yes / 2=No)
    df["alive_independent"] = (pd.to_numeric(raw["aliveind6"], errors="coerce") == 1).astype(float)
    df["treated"] = (pd.to_numeric(raw["itt_treat"], errors="coerce") == 0).astype(float)  # 1 = alteplase
    df["source"] = 1
    return df


def load_ist1(path):
    raw = pd.read_csv(path, low_memory=False)
    df = pd.DataFrame()
    df["age"]             = pd.to_numeric(raw["AGE"], errors="coerce")
    df["delay_hours"]     = pd.to_numeric(raw["RDELAY"], errors="coerce")
    df["sbp"]             = pd.to_numeric(raw["RSBP"], errors="coerce")
    df["male"]            = raw["SEX"].map({"M": 1, "F": 0})
    df["atrial_fib"]      = raw["RATRIAL"].map({"Y": 1, "N": 0})     # NaN for the 984 pilot patients
    df["visible_infarct"] = raw["RVISINF"].map({"Y": 1, "N": 0})
    # IST-1's OWN 6-month functional outcome, via OCCODE (1=dead, 2=dependent,
    # 3=not recovered, 4=recovered) -> alive & independent = OCCODE in {3, 4}.
    # HELD OUT of the transport: every model fits on IST-3 (source==1) only, so this
    # never enters membership/weights/outcome/propensity/gamma. It exists purely to
    # validate the IST-3 outcome model's predictions on the IST-1 population.
    df["alive_independent"] = pd.to_numeric(raw["OCCODE"], errors="coerce").map({1: 0, 2: 0, 3: 1, 4: 1})
    df["treated"] = float("nan")     # IST-1 has no alteplase arm; only IST-3 defines treatment
    df["source"] = 0
    return df


def main():
    ist3 = load_ist3(IST3_PATH)
    ist1 = load_ist1(IST1_PATH)
    combined = pd.concat([ist3[SHARED], ist1[SHARED]], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"IST-3 rows: {len(ist3)}   IST-1 rows: {len(ist1)}   combined: {len(combined)}")
    print("\nsource counts (1=IST-3, 0=IST-1):")
    print(combined["source"].value_counts())
    print("\nmissing values per column:")
    print(combined.isna().sum())
    print("\nfirst rows:")
    print(combined.head())
    print(f"\nsaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()