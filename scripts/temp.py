import glob
from pathlib import Path

def force_zero_class(label_dir):
    print(f"Scanning {label_dir} for negative Class IDs...")
    txt_files = glob.glob(f"{label_dir}/**/*.txt", recursive=True)
    fixed_count = 0

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        needs_fixing = False

        for line in lines:
            parts = line.strip().split()
            # If the line has data (Class + X Y W H)
            if len(parts) == 5:
                # If the class ID is NOT 0 (e.g. -1)
                if parts[0] != "0":
                    parts[0] = "0"  # Force it to be 0
                    needs_fixing = True
                
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        # Overwrite the file only if we fixed a class ID
        if needs_fixing:
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
            fixed_count += 1

    print(f"✅ Fixed {fixed_count} label files in {label_dir}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.absolute()
    train_dir = project_root / "data/small_bird_dataset/labels/train"
    val_dir = project_root / "data/small_bird_dataset/labels/val"
    
    force_zero_class(str(train_dir))
    force_zero_class(str(val_dir))