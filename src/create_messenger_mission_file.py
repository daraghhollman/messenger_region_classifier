import datetime as dt
import pickle

import hermpy.mag


def main():
    save_mission("./data/mission_file.pkl")


def chunk_dates(start_date, end_date, days_per_chunk):
    """Divide a date range into smaller chunks of a specified size in days."""
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + dt.timedelta(days=days_per_chunk), end_date)
        yield current_start, current_end
        current_start = current_end


def save_mission(path: str, days_per_chunk=60):
    mission_start = dt.datetime(2011, 3, 23, 15, 37)
    mission_end = dt.datetime(2015, 4, 30, 15, 8)

    with open(path, "wb") as f:
        for start_date, end_date in chunk_dates(
            mission_start, mission_end, days_per_chunk
        ):  # Process 30 days at a time
            data_chunk = hermpy.mag.Load_Between_Dates(
                "./data/messenger/one_minute_avg/",
                start_date,
                end_date,
                strip=True,
                aberrate=True,
            )
            hermpy.mag.Remove_Spikes(data_chunk)

            # Reduce columns to save on both storage size,
            # and when loaded into memory.
            columns_to_include = [
                "date",
                "|B|",
                "Bx'",
                "By'",
                "Bz'",
                "X MSM' (radii)",
                "Y MSM' (radii)",
                "Z MSM' (radii)",
            ]
            data_chunk = data_chunk[columns_to_include]

            print(f"Dumped data between {start_date}, and {end_date}")
            pickle.dump(data_chunk, f)


if __name__ == "__main__":
    main()
