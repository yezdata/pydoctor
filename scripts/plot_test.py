import os
import glob
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    libs_dir = "data/libs"
    py_files = glob.glob(os.path.join(libs_dir, "**", "*.py"), recursive=True)

    char_counts = []
    for file_path in py_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                char_counts.append(len(content))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not char_counts:
        print("No .py files found")
        return

    non_empty = [c for c in char_counts if c > 0]
    if not non_empty:
        print("All .py files are empty.")
        return

    non_empty.sort()
    print(f"Total .py files: {len(char_counts)}")
    print(f"Empty files: {len(char_counts) - len(non_empty)}")
    print(f"Min: {non_empty[:10]}")
    print(f"Max: {non_empty[-10:][::-1]}")
    cutoff = int(len(non_empty) * 0.98) if len(non_empty) > 50 else len(non_empty)

    plot_counts = non_empty[:cutoff]

    plt.figure(figsize=(10, 6))
    plt.hist(plot_counts, bins=50, color="skyblue", edgecolor="black")
    plt.title("Distribution of Character Counts in .py Files")
    plt.xlabel("Character Count")
    plt.ylabel("Number of Files")
    plt.grid(axis="y", alpha=0.75)

    mean_val = statistics.mean(plot_counts)
    median_val = statistics.median(plot_counts)
    plt.axvline(
        mean_val,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_val:.0f}",
    )
    plt.axvline(
        median_val,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: {median_val:.0f}",
    )
    plt.legend()

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(20))
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = "char_count_distribution.png"
    plt.savefig(output_file)


if __name__ == "__main__":
    main()
