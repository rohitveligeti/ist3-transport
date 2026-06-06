"""
Preprocessing: combine IST-3 and IST-1 into one dataset with a shared,
all-numeric covariate encoding.

Six shared covariates (same numeric encoding in both):
    age              years
    delay_hours      hours from stroke onset to randomisation
    sbp              systolic BP at randomisation (mmHg)
    male             1 = male, 0 = female
    atrial_fib       1 = yes, 0 = no
    visible_infarct  1 = yes, 0 = no
Plus:
    source             1 = IST-3, 0 = IST-1
    alive_independent  1 = alive & independent (OHS 0-2) at 6 months, 0 = not;
                       filled for IST-3 only (NaN for IST-1) per request.
                       IST-1 actually has this too (OCCODE in {3,4}); see the
                       commented line in load_ist1 to turn it on.
"""

import pandas as pd

IST3_PATH = "df.sas7bdat"
IST1_PATH = "ist1.CSV"
OUTPUT_PATH = "ist_combined.csv"

SHARED = ["source", "age", "delay_hours", "sbp", "male",
          "atrial_fib", "visible_infarct", "alive_independent"]


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
    # not populated per request:
    df["alive_independent"] = float("nan")
    # To fill it from IST-1 instead, comment the line above and use:
    # df["alive_independent"] = raw["OCCODE"].map({1: 0, 2: 0, 3: 1, 4: 1})  # OHS 0-2 analog
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