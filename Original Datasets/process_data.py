import json
import os
import pandas as pd
from rich.console import Console

console = Console()

os.makedirs("Processed Datasets", exist_ok=True)

# -----------------------------
# Load the JSON dataset
# -----------------------------
with open("Original Datasets/intents.json", "r") as file:
    data = json.load(file)

# -----------------------------
# Convert JSON into tabular format
# One row per (pattern, tag) pair. Keep ALL responses per tag
# (not just responses[0]) so the bot can vary its replies later.
# -----------------------------
rows = []

for intent in data["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        rows.append({
            "tag": tag,
            "pattern": pattern.strip(),
            "response": intent["responses"][0]  # primary response for tabular view
        })

df = pd.DataFrame(rows)

# -----------------------------
# Data Quality Checks
# -----------------------------
console.rule("[bold cyan]Dataset Overview[/bold cyan]")
console.print(f"Shape: [bold bright_white]{df.shape[0]} rows x {df.shape[1]} columns[/bold bright_white]")

console.print("\n[bold cyan]Tags and counts:[/bold cyan]")
for tag, count in df["tag"].value_counts().items():
    console.print(f"  {tag:<25} [bold bright_white]{count}[/bold bright_white]")

console.rule("[bold cyan]Missing Values[/bold cyan]")
for col, missing in df.isnull().sum().items():
    if missing > 0:
        console.print(f"  {col}: [bold red]{missing}[/bold red]")
    else:
        console.print(f"  {col}: [bold green]0[/bold green]")

console.rule("[bold cyan]Duplicate Check[/bold cyan]")
full_dup_count = df.duplicated().sum()
console.print("Exact duplicate rows (all features combined):")
if full_dup_count > 0:
    console.print(f"  [bold red]{full_dup_count}[/bold red]")
else:
    console.print(f"  [bold green]0[/bold green]")

# Drop exact duplicate (tag, pattern) pairs if any slipped in
before = len(df)
df = df.drop_duplicates(subset=["tag", "pattern"]).reset_index(drop=True)
dropped = before - len(df)
if dropped > 0:
    console.print(f"\n[bold yellow]Dropped {dropped} duplicate pattern rows[/bold yellow]")
else:
    console.print(f"\n[bold green]Dropped 0 duplicate pattern rows[/bold green]")

# Flag any empty/whitespace-only patterns (shouldn't exist, but check)
empty_patterns = df[df["pattern"].str.strip() == ""]
if len(empty_patterns) > 0:
    console.print(f"[bold yellow]WARNING: {len(empty_patterns)} empty patterns found[/bold yellow]")
    df = df[df["pattern"].str.strip() != ""]

# -----------------------------
# Save processed dataset (no derived length features - they add
# no signal for intent classification and just clutter the schema)
# -----------------------------
df.to_csv("Processed Datasets/processed_data.csv", index=False)

columns = pd.DataFrame(df.columns, columns=["Column_Name"])
columns.to_csv("Processed Datasets/processed_data_columns.csv", index=False)

console.rule("[bold cyan]Saved[/bold cyan]")
console.print("[bold green]Final dataset saved: [underline cyan]Processed Datasets/processed_data.csv[/]")
console.print(f"Columns: [bold bright_white]{df.columns.tolist()}[/bold bright_white]")

# -----------------------------
# Separate EDA export with length features (not used for training -
# kept out of processed_data.csv since word count carries no
# classification signal, but useful for exploratory analysis/slides)
# -----------------------------
df_eda = df.copy()
df_eda["pattern_length"] = df_eda["pattern"].apply(lambda x: len(x.split()))
df_eda["response_length"] = df_eda["response"].apply(lambda x: len(x.split()))
df_eda.to_csv("Processed Datasets/processed_data_eda.csv", index=False)

console.print("[bold green]EDA dataset saved: [underline cyan]Processed Datasets/processed_data_eda.csv[/]")
console.print(f"Columns: [bold bright_white]{df_eda.columns.tolist()}[/bold bright_white]")