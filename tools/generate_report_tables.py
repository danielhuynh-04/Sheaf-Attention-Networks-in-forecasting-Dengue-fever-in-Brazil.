import pandas as pd
import os


def generate_summary():

    input_file = "evaluations/final_results.csv"
    output_file = "evaluations/report_table.csv"

    if not os.path.exists(input_file):
        print("No benchmark file found.")
        return

    df = pd.read_csv(input_file)

    summary = df.groupby("Model").agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        Time_mean=("TrainTime", "mean")
    )

    summary.to_csv(output_file)

    print("Report table generated at:", output_file)
    print(summary)


if __name__ == "__main__":
    generate_summary()
