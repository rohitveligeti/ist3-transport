import pandas as pd

if __name__ == "__main__":
    sas_file = "df.sas7bdat"
    csv_file = "df.csv"

    df = pd.read_sas(sas_file, format="sas7bdat", encoding="utf-8")
    country_counts = df['country'].value_counts(dropna=False)
    print(country_counts)
    df.to_csv(csv_file, index=False)
    print(f"Saved {sas_file} to {csv_file}")

