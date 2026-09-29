import csv
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


def write_csv(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("Name", "Path", "KL"))
        writer.writerows(rows)


def generate_csv() -> list[tuple[str, str, int]]:
    project_dir = Path(__file__).resolve().parent
    folder = project_dir / "data/archive/OSAIL_KL_Dataset/Labeled"
    output_dir = project_dir / "data/csv"

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    rows = [
        (file.name, str(file), int(file.parent.name))
        for file in sorted(folder.rglob("*"))
        if file.is_file()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "files.csv", rows)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [row[2] for row in rows]
    for fold, (train_indices, val_indices) in enumerate(cv.split(rows, labels)):
        write_csv(
            output_dir / f"train_fold_{fold}.csv", [rows[i] for i in train_indices]
        )
        write_csv(output_dir / f"val_fold_{fold}.csv", [rows[i] for i in val_indices])

    return rows


if __name__ == "__main__":
    generate_csv()
