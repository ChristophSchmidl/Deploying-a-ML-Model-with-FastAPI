import os
import pandas as pd


def show_data_head(data_path):
    df = pd.read_csv(data_path)
    print(df.head())

def remove_trailing_whitespaces(src_data_path, dst_data_path):
    df = pd.read_csv(src_data_path)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.columns = df.columns.str.strip()

    # Write the cleaned data back to the CSV file
    df.to_csv(dst_data_path, index=False)

if __name__ == '__main__':
    messy_df_path = os.path.join(os.getcwd(), "data", "census.csv")
    clean_df_path = os.path.join(os.getcwd(), "data", "census_clean.csv")

    show_data_head(messy_df_path)
    remove_trailing_whitespaces(messy_df_path, clean_df_path)
    show_data_head(messy_df_path)

