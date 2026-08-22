"""Interactive YOLO dataset management tool.

Layout: dataset/{train,val,test}/{images,labels} and data.yaml
Installation: py -m pip install questionary PyYAML
Run: py merge_yolo_datasets.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import math
import os
import random
import shutil
import subprocess
import sys

try:
    import yaml
    import questionary
    import questionary.prompts.common as questionary_common
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
except ImportError as exc:
    raise SystemExit("Installation: py -m pip install questionary PyYAML") from exc

BASE_DIRECTORY = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DATA_YAML_NAME = "data.yaml"
DEFAULT_RATIOS = (80.0, 10.0, 10.0)
SEED = 42
CREATE_BACKUP = True

# Show exactly "*" at the beginning of selected checkbox folders.
questionary_common.INDICATOR_SELECTED = "*"
questionary_common.INDICATOR_UNSELECTED = " "

# Set before by configure_platform() at program startup.
SELECTED_OS = None
SELECTED_LINUX_DISTRO = None


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path
    source_split: str
    class_ids: frozenset[int]
    box_counts: Counter


class BackMenu(Exception):
    """Signal that the user wants to return to the previous menu."""


def with_back_choice(choices):
    """Append a visible Back option to menu choices."""
    values = []
    for choice in choices:
        if isinstance(choice, questionary.Choice):
            values.append(choice)
        else:
            values.append(choice)
    values.append(questionary.Choice("(Back)", "__BACK__"))
    return values


def ask_text(message, default=""):
    value = questionary.text(message, default=default).ask()
    if value is None:
        raise KeyboardInterrupt
    return value.strip()


def ask_confirm(message, default=False):
    value = questionary.confirm(message, default=default).ask()
    if value is None:
        raise KeyboardInterrupt
    return bool(value)


def ask_select(message, choices):
    # Prevent questionary from adding the numbers we already wrote in choice labels a second time.
    value = questionary.select(
        message, choices=with_back_choice(choices), use_shortcuts=False
    ).ask()
    if value is None:
        raise KeyboardInterrupt
    if value == "__BACK__":
        raise BackMenu
    return value


def ask_checkbox(message, choices):
    value = questionary.checkbox(
        message, choices=with_back_choice(choices),
        instruction="(Arrow keys: navigate, Space: select/deselect, Enter: confirm)",
    ).ask()
    if value is None:
        raise KeyboardInterrupt
    if "__BACK__" in value:
        raise BackMenu
    return value


def normalized_name(value):
    return " ".join(str(value).strip().casefold().split())


def show_progress(title, current, total, detail="", finish=False):
    """Single-line percentage progress for Windows and Linux terminals."""
    percentage = 100.0 if total <= 0 else min(100.0, current * 100.0 / total)
    bar_width = 30
    filled = round(bar_width * percentage / 100.0)
    bar = "#" * filled + "-" * (bar_width - filled)
    suffix = f" | {detail}" if detail else ""
    print(
        f"\r{title}: [{bar}] %{percentage:6.2f}{suffix}",
        end="\n" if finish or current >= total else "",
        flush=True,
    )


def configure_platform():
    """Select the operating system before for use throughout the program."""
    global SELECTED_OS, SELECTED_LINUX_DISTRO

    while True:
        SELECTED_OS = ask_select(
            "Which operating system are you using?",
            [questionary.Choice("(1) Windows", "windows"),
             questionary.Choice("(2) Linux", "linux")],
        )

        if SELECTED_OS == "linux":
            try:
                SELECTED_LINUX_DISTRO = ask_select(
                    "Your Linux distribution:",
                    [questionary.Choice("(1) Arch / Arch-based", "arch"),
                     questionary.Choice("(2) Debian / Ubuntu-based", "debian"),
                     questionary.Choice("(3) Fedora", "fedora")],
                )
            except BackMenu:
                # Linux distro menu -> Windows/Linux menu
                continue
            break

        # Windows has no intermediate distro menu.
        SELECTED_LINUX_DISTRO = None
        break

    detected = "windows" if os.name == "nt" else "linux" if sys.platform.startswith("linux") else sys.platform
    if detected != SELECTED_OS:
        print(
            f"WARNING: The Python environment {detected!r} appears to be running as "
            f"{SELECTED_OS!r} but you selected. OS-specific operations will follow your selection."
        )


def split_dirs(root, create=False):
    result = {}
    for split in SPLITS:
        split_name = split
        # Read legacy Roboflow folders; newly created folders always use "val".
        if split == "val" and not (root / "val").exists() and (root / "valid").exists():
            split_name = "valid"
        images, labels = root / split_name / "images", root / split_name / "labels"
        if create:
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
        result[split] = {"images": images, "labels": labels}
    return result


def looks_like_dataset(root):
    return root.is_dir() and any(
        (root / s / "images").is_dir() or (root / s / "labels").is_dir()
        for s in (*SPLITS, "valid")
    )


def dataset_folders():
    return sorted(
        (p for p in BASE_DIRECTORY.iterdir() if looks_like_dataset(p)),
        key=lambda p: p.name.casefold(),
    )


def choose_one_dataset(message, exclude=None):
    excluded = {p.resolve() for p in (exclude or set())}
    folder = choose_directory(message, BASE_DIRECTORY).resolve()
    if folder in excluded:
        raise ValueError("This folder cannot be reused as a source or destination in this operation.")
    if not looks_like_dataset(folder):
        raise ValueError(
            "The selected folder is not a dataset containing images or labels under train/val/test: "
            f"{folder}"
        )
    return folder


def choose_destination_folder(message, exclude=None):
    """Select an empty or existing destination folder using the terminal navigator."""
    excluded = {p.resolve() for p in (exclude or set())}
    folder = choose_directory(message, BASE_DIRECTORY).resolve()
    if folder in excluded:
        raise ValueError("A folder selected as a source cannot be used as the destination.")
    return folder


def choose_many_datasets(message, exclude=None, allow_empty=False):
    """Select one or more dataset folders from different directories."""
    excluded = {p.resolve() for p in (exclude or set())}
    selected = []
    if allow_empty and not ask_confirm(f"{message} Select any folders?", False):
        return selected
    while True:
        folder = choose_one_dataset(message, excluded | {p.resolve() for p in selected})
        selected.append(folder)
        print(f"Selected ({len(selected)}): {folder}")
        if not ask_confirm("Add another dataset folder to this group?", False):
            return selected


def choose_ordered_datasets(message, available=None):
    """Select sources from different directories while preserving selection order."""
    print(
        "Classes in the selected folders' data.yaml files are used in the main data.yaml; "
        "folder names are not used as class names."
    )
    print("Selection order determines the class order in the main data.yaml.")
    return choose_many_datasets(message, allow_empty=False)


def choose_directory(message, start=None):
    """Select a destination directory by navigating through folders in the terminal."""
    current = Path(start or BASE_DIRECTORY).resolve()
    cursor = 0
    bindings = KeyBindings()

    def subdirectories():
        try:
            folders = list(p for p in current.iterdir() if p.is_dir())
            if os.name == "nt" and current.parent == current:
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    drive = Path(f"{letter}:\\")
                    try:
                        if drive.exists() and drive.resolve() != current:
                            folders.append(drive)
                    except OSError:
                        continue
            unique = {str(folder.resolve()).casefold(): folder for folder in folders}
            return sorted(unique.values(), key=lambda p: str(p).casefold())
        except OSError:
            return []

    def render():
        folders = subdirectories()
        tokens = [
            ("class:question", f"{message}\n"),
            ("class:instruction", f"Current directory: {current}\n"),
            ("class:instruction",
             "Up/Down: navigate | Right/Space: enter folder | Left/Esc: parent directory | "
             "Enter: select this directory | Ctrl+C: cancel\n\n"),
        ]
        if not folders:
            tokens.append(("class:instruction", "  (No subfolders or permission to read)\n"))
        for index, folder in enumerate(folders):
            pointer = ">" if index == cursor else " "
            style = "class:selected" if index == cursor else "class:text"
            tokens.append((style, f"{pointer} {folder.name}/\n"))
        return FormattedText(tokens)

    control = FormattedTextControl(render)

    @bindings.add("up")
    @bindings.add("k")
    def move_up(event):
        nonlocal cursor
        folders = subdirectories()
        if folders:
            cursor = (cursor - 1) % len(folders)
        event.app.invalidate()

    @bindings.add("down")
    @bindings.add("j")
    def move_down(event):
        nonlocal cursor
        folders = subdirectories()
        if folders:
            cursor = (cursor + 1) % len(folders)
        event.app.invalidate()

    @bindings.add("right")
    @bindings.add("space")
    def enter_directory(event):
        nonlocal current, cursor
        folders = subdirectories()
        if folders:
            current = folders[min(cursor, len(folders) - 1)].resolve()
            cursor = 0
        event.app.invalidate()

    @bindings.add("left")
    @bindings.add("escape")
    def parent_directory(event):
        nonlocal current, cursor
        parent = current.parent
        if parent != current:
            current = parent
            cursor = 0
        event.app.invalidate()

    @bindings.add("enter")
    def accept(event):
        event.app.exit(result=current)

    @bindings.add("c-c")
    def cancel(event):
        event.app.exit(exception=KeyboardInterrupt())

    application = Application(
        layout=Layout(Window(control, always_hide_cursor=True)),
        key_bindings=bindings,
        full_screen=False,
    )
    return application.run()


def image_files(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def open_path(path):
    try:
        if SELECTED_OS == "windows":
            if os.name != "nt":
                raise RuntimeError("The Windows image-opening operation cannot be run outside Windows.")
            os.startfile(str(path))
        elif SELECTED_OS == "linux":
            if shutil.which("xdg-open") is None:
                install_commands = {
                    "arch": "sudo pacman -S xdg-utils",
                    "debian": "sudo apt install xdg-utils",
                    "fedora": "sudo dnf install xdg-utils",
                }
                command = install_commands.get(SELECTED_LINUX_DISTRO, "install the xdg-utils package")
                raise RuntimeError(f"xdg-open not found. Kurulum: {command}")
            subprocess.Popen(["xdg-open", str(path)])
        else:
            raise RuntimeError("Operating system has not been selected.")
    except Exception as exc:
        print(f"Could not open image: {exc}")


def parse_label(label, known_ids=None):
    lines, counts = [], Counter()
    text = label.read_text(encoding="utf-8-sig")
    for line_no, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid YOLO line: {label}:{line_no}\n{raw}") from exc
        if class_id < 0:
            raise ValueError(f"Negative class ID: {label}:{line_no}")
        if known_ids is not None and class_id not in known_ids:
            raise ValueError(f"Class ID not present in data.yaml {class_id}: {label}:{line_no}")
        is_box = len(parts) == 5
        is_polygon = len(coords) >= 6 and len(coords) % 2 == 0
        if not (is_box or is_polygon):
            raise ValueError(f"Invalid box/polygon format: {label}:{line_no}")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
            raise ValueError(f"Coordinate outside 0-1 range: {label}:{line_no}")
        if is_box and (coords[2] <= 0 or coords[3] <= 0):
            raise ValueError(f"Invalid box dimensions: {label}:{line_no}")
        lines.append((class_id, parts[1:]))
        counts[class_id] += 1
    return lines, counts


def load_yaml(root, required=True):
    path = root / DATA_YAML_NAME
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"data.yaml not found: {path}")
        return {}, {}, path
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    names = data.get("names")
    if isinstance(names, list):
        mapping = {i: str(v).strip() for i, v in enumerate(names)}
    elif isinstance(names, dict):
        mapping = {int(i): str(v).strip() for i, v in names.items()}
    else:
        raise ValueError(f"No valid names found in data.yaml: {path}")
    duplicate = [n for n, c in Counter(normalized_name(v) for v in mapping.values()).items() if c > 1]
    if not mapping or any(not v for v in mapping.values()) or duplicate:
        raise ValueError(f"Empty or duplicate class name: {path}; again={duplicate}")
    return data, mapping, path


def write_yaml(path, data, names):
    output = dict(data)
    output["names"] = {i: names[i] for i in sorted(names)}
    output["nc"] = len(names)
    output["train"], output["val"], output["test"] = (
        "train/images", "val/images", "test/images"
    )
    output.pop("path", None)
    path.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def ensure_structure_confirmation():
    if not ask_confirm(
        'SPLITS=("train", "val", "test"). Does each dataset folder contain '
        "split/images and split/labels duzenini validates musunuz?", True
    ):
        raise SystemExit("Fix the folder structure and run the program again.")


def repair_missing_pairs(root):
    dirs = split_dirs(root, create=True)
    for split, paths in dirs.items():
        images = image_files(paths["images"])
        grouped = defaultdict(list)
        for image in images:
            grouped[image.stem].append(image)
        duplicate = {stem: files for stem, files in grouped.items() if len(files) > 1}
        if duplicate:
            stem, files = next(iter(duplicate.items()))
            raise RuntimeError(f"same stem adli more than one more image: {stem} -> {files}")
        for image in list(images):
            label = paths["labels"] / f"{image.stem}.txt"
            if label.is_file():
                continue
            action = ask_select(
                f"{image.name} label for ({root.name}/{split}).",
                [questionary.Choice("(1) Open image", "open"),
                 questionary.Choice("(2) Devam et / simdilik atla", "continue"),
                 questionary.Choice("(3) Programi sonlandir", "exit")],
            )
            if action == "exit":
                raise SystemExit("Terminated because of a missing label.")
            if action == "continue":
                continue
            open_path(image)
            resolution = ask_select(
                f"{image.name} What should be done for",
                [questionary.Choice("(1) Delete image", "delete"),
                 questionary.Choice("(2) empty label create", "empty"),
                 questionary.Choice("(3) Do nothing and finish", "exit")],
            )
            if resolution == "delete":
                image.unlink()
            elif resolution == "empty":
                label.touch()
            else:
                raise SystemExit("Terminated at the user's request.")
        stems = {p.stem for p in image_files(paths["images"])}
        orphans = [p for p in paths["labels"].glob("*.txt") if p.stem not in stems]
        if orphans:
            raise RuntimeError("Labels without corresponding images:\n" +
                               "\n".join(map(str, orphans[:20])))


def manifest(root, require_yaml=False, validate_class_ids=True):
    _, names, _ = load_yaml(root, required=require_yaml)
    known = set(names) if names and validate_class_ids else None
    result = []
    items = [
        (split, paths, image)
        for split, paths in split_dirs(root, create=True).items()
        for image in image_files(paths["images"])
    ]
    show_progress("Dataset scan", 0, len(items), f"0/{len(items)} image")
    for index, (split, paths, image) in enumerate(items, 1):
            label = paths["labels"] / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Label not found: {label}")
            _, counts = parse_label(label, known)
            result.append(Pair(image, label, split, frozenset(counts), counts))
            if index == len(items) or index % max(1, len(items) // 100) == 0:
                show_progress(
                    "Dataset scan", index, len(items), f"{index}/{len(items)} image",
                    finish=index == len(items),
                )
    return result


def backup_metadata(root, purpose):
    if not CREATE_BACKUP:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = root.parent / f"{root.name}_{purpose}_backup_{stamp}.zip"
    yml = root / DATA_YAML_NAME
    files = ([yml] if yml.is_file() else []) + [
        label
        for paths in split_dirs(root).values()
        if paths["labels"].is_dir()
        for label in sorted(paths["labels"].glob("*.txt"))
    ]
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        show_progress("Backup", 0, len(files), f"0/{len(files)} file")
        for index, file in enumerate(files, 1):
            archive.write(file, file.relative_to(root))
            if index == len(files) or index % max(1, len(files) // 100) == 0:
                show_progress(
                    "Backup", index, len(files), f"{index}/{len(files)} file",
                    finish=index == len(files),
                )
    return target


def unique_stem(stem, source_name, image_dir, label_dir):
    existing = {p.stem for p in image_files(image_dir)} | {
        p.stem for p in label_dir.glob("*.txt")
    }
    if stem not in existing:
        return stem, False
    base = f"{'_'.join(source_name.split())}__{stem}"
    candidate, counter = base, 1
    while candidate in existing:
        candidate = f"{base}_{counter:03d}"
        counter += 1
    return candidate, True


def rewrite_label(lines, id_map):
    output = []
    for old_id, coords in lines:
        if old_id not in id_map:
            raise ValueError(f"No target mapping for class ID: {old_id}")
        output.append(" ".join([str(id_map[old_id]), *coords]))
    return "\n".join(output) + ("\n" if output else "")


def merge_classes_within_dataset(root, data, names, yaml_path):
    selected = set(ask_checkbox(
        "to be merged into one class classes select:",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(names)],
    ))
    if len(selected) < 2:
        raise ValueError("At least two classes must be selected for merging.")
    target_id = ask_select(
        "Under which target class name should the selected classes be merged?",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(selected)],
    )

    # The target class remains in its old position; the other merged classes are removed.
    retained_old_ids = [
        cid for cid in sorted(names) if cid not in selected or cid == target_id
    ]
    retained_to_new = {old: new for new, old in enumerate(retained_old_ids)}
    id_map = {}
    for old_id in sorted(names):
        if old_id in selected:
            id_map[old_id] = retained_to_new[target_id]
        else:
            id_map[old_id] = retained_to_new[old_id]
    new_names = {retained_to_new[old]: names[old] for old in retained_old_ids}

    pairs = manifest(root, require_yaml=True)
    changed_labels = merged_boxes = empty_labels = 0
    plans = []
    selected_box_counts = Counter()
    show_progress("Class merge scan", 0, len(pairs), f"0/{len(pairs)} label")
    for index, pair in enumerate(pairs, 1):
        lines, counts = parse_label(pair.label, set(names))
        selected_box_counts.update({cid: counts[cid] for cid in selected})
        merged_boxes += sum(counts[cid] for cid in selected if cid != target_id)
        empty_labels += int(not lines)
        new_text = rewrite_label(lines, id_map)
        old_text = pair.label.read_text(encoding="utf-8-sig")
        if new_text != old_text:
            plans.append((pair.label, new_text))
            changed_labels += 1
        if index == len(pairs) or index % max(1, len(pairs) // 100) == 0:
            show_progress(
                "Class merge scan", index, len(pairs), f"{index}/{len(pairs)} label",
                finish=index == len(pairs),
            )

    print("\nClass merge plan:")
    for cid in sorted(selected):
        print(
            f"  {cid}:{names[cid]} -> "
            f"{retained_to_new[target_id]}:{names[target_id]} "
            f"(box={selected_box_counts[cid]})"
        )
    print("\nNew data.yaml class order:")
    for cid in sorted(new_names):
        print(f"  {cid}: {new_names[cid]}")
    print(
        f"Labels to change={changed_labels}, boxes to convert to the target class={merged_boxes}, "
        f"empty/negative labels={empty_labels}"
    )
    if not ask_confirm("Apply the class merge plan?", False):
        print("No changes were made.")
        return

    backup = backup_metadata(root, "class_merge")
    if backup:
        print("Label/YAML backup:", backup)
    show_progress("Class merge", 0, len(plans), f"0/{len(plans)} label")
    for index, (label, text) in enumerate(plans, 1):
        label.write_text(text, encoding="utf-8")
        if index == len(plans) or index % max(1, len(plans) // 100) == 0:
            show_progress(
                "Class merge", index, len(plans), f"{index}/{len(plans)} label",
                finish=index == len(plans),
            )
    write_yaml(yaml_path, data, new_names)
    validate_dataset(root, interactive=False, require_yaml=True)
    print(
        f"Class merge completed: {', '.join(names[c] for c in sorted(selected))} "
        f"-> {names[target_id]}"
    )


def force_all_labels_to_class_id(root, data, names, yaml_path):
    if names:
        new_id = ask_select(
            "Which class should all non-empty label lines be converted to?",
            [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(names)],
        )
        target_name = names[new_id]
    else:
        print("No selectable class was found in data.yaml; you must enter the target manually.")
        raw_id = ask_text("Butun dolu label satirlarinin new class ID value:")
        try:
            new_id = int(raw_id)
        except ValueError as exc:
            raise ValueError("Class ID must be an integer greater than or equal to 0.") from exc
        if new_id < 0:
            raise ValueError("Class ID cannot be negative.")
        target_name = ask_text("Class name for this target ID:")
        if not target_name:
            raise ValueError("Target class name cannot be empty.")
    if new_id != 0:
        print(
            "WARNING: The training ID of a single-class dataset should normally be 0. "
            f"{new_id} value may be used as preparation before merging into the main dataset; "
            "do not train this intermediate dataset directly."
        )

    # The purpose of this mode is to repair broken/legacy IDs. Therefore, the old ID does not
    # need to be defined in data.yaml; line format and coordinates
    # are still fully validated.
    pairs = manifest(root, require_yaml=False, validate_class_ids=False)
    plans = []
    box_count = empty_count = already_correct = 0
    old_id_counts = Counter()
    for pair in pairs:
        lines, counts = parse_label(pair.label, known_ids=None)
        old_id_counts.update(counts)
        box_count += len(lines)
        empty_count += int(not lines)
        new_text = rewrite_label(lines, {old_id: new_id for old_id in counts})
        old_text = pair.label.read_text(encoding="utf-8-sig")
        if new_text != old_text:
            plans.append((pair.label, new_text))
        else:
            already_correct += int(bool(lines))

    print("\nBulk class ID replacement plan:")
    for old_id in sorted(old_id_counts):
        old_name = names.get(old_id, "<not defined in data.yaml>")
        print(f"  {old_id}:{old_name} -> {new_id}:{target_name} (box={old_id_counts[old_id]})")
    print(
        f"Total boxes={box_count}, labels to change={len(plans)}, "
        f"already-correct non-empty labels={already_correct}, empty/negative labels={empty_count}"
    )
    if not ask_confirm("Change all non-empty label class IDs?", False):
        print("No changes were made.")
        return

    backup = backup_metadata(root, "class_id")
    if backup:
        print("Label/YAML backup:", backup)
    show_progress("Change label IDs", 0, len(plans), f"0/{len(plans)} label")
    for index, (label, text) in enumerate(plans, 1):
        label.write_text(text, encoding="utf-8")
        if index == len(plans) or index % max(1, len(plans) // 100) == 0:
            show_progress(
                "Change label IDs", index, len(plans), f"{index}/{len(plans)} label",
                finish=index == len(plans),
            )
    write_yaml(yaml_path, data, {new_id: target_name})
    validate_dataset(root, interactive=False, require_yaml=True)
    print(
        f"Bulk class ID replacement completed. All non-empty boxes: "
        f"{new_id}:{target_name}"
    )


def extract_or_delete_class_pairs(root, data, names):
    selected = set(ask_checkbox(
        "Select the classes to filter:",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(names)],
    ))
    if not selected:
        raise ValueError("At least one class must be selected.")

    policy = ask_select(
        "How should images be filtered?",
        [questionary.Choice(
            "An object whose class will not exist in the new data.yaml may remain in an image used to train the selected class "
            "training (recommended)",
            "allow_other"),
         questionary.Choice(
            "Use only images containing boxes from the selected classes",
            "only_effective")],
    )
    action = ask_select(
        "Choose the operation:",
        [questionary.Choice("Copy to another dataset", "copy"),
         questionary.Choice("Delete", "delete")],
    )

    destination = None
    if action == "copy":
        destination_parent = choose_directory(
            "Select the parent directory where the copy will be saved:", BASE_DIRECTORY
        )
        destination_name = ask_text(
            "New destination dataset folder name "
            "(if left blank train/val/test and data.yaml inside the selected directory "
            "is used or created):"
        )
        destination = (
            destination_parent / validate_new_dataset_name(destination_name)
            if destination_name else destination_parent
        ).resolve()
        try:
            destination.resolve().relative_to(root.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("destination cannot be inside the source dataset.")

    scope = ask_select(
        "What should this apply to:",
        [questionary.Choice("Selected classes", "selected"),
         questionary.Choice("Unselected classes", "unselected")],
    )
    effective_ids = selected if scope == "selected" else set(names) - selected
    if not effective_ids:
        raise ValueError("No usable classes remain after this selection.")

    # The filtered dataset uses consecutive IDs starting from 0.
    filtered_order = [cid for cid in sorted(names) if cid in effective_ids]
    filtered_id_by_old = {old_id: new_id for new_id, old_id in enumerate(filtered_order)}
    destination_data = dict(data)
    destination_names = {}
    destination_yaml = destination / DATA_YAML_NAME if destination is not None else None
    copy_id_map = dict(filtered_id_by_old)

    if action == "copy":
        if (destination / "images").is_dir() or (destination / "labels").is_dir():
            raise RuntimeError(
                "selected destination images/train + labels/train ZIP in the layout is visible. "
                "This filtering operation uses the train/images + train/labels working layout."
            )
        yaml_exists = destination_yaml.is_file()
        if yaml_exists:
            destination_data, destination_names, _ = load_yaml(destination, required=True)
            output_names = dict(destination_names)
            target_by_name = {
                normalized_name(class_name): class_id
                for class_id, class_name in output_names.items()
            }
            occupied_ids = set(output_names)
            copy_id_map = {}

            for old_id in filtered_order:
                filtered_id = filtered_id_by_old[old_id]
                class_name = names[old_id]
                normalized = normalized_name(class_name)
                if normalized in target_by_name:
                    target_id = target_by_name[normalized]
                    if target_id != filtered_id and not ask_confirm(
                        f"The data.yaml to be copied contains {filtered_id}:{class_name}, destination "
                        f"data.yaml contains {target_id}:{output_names[target_id]}. "
                        "To preserve the dataset to be copied label class ID value "
                        f"{target_id}? If No, the operation is cancelled.",
                        True,
                    ):
                        raise SystemExit("Class ID mapping was rejected; copying was cancelled.")
                    copy_id_map[old_id] = target_id
                    continue

                if filtered_id not in occupied_ids:
                    target_id = filtered_id
                else:
                    conflicting_name = output_names[filtered_id]
                    candidates = range(max(occupied_ids, default=-1) + 2)
                    target_id = min(
                        (candidate for candidate in candidates if candidate not in occupied_ids),
                        key=lambda candidate: (abs(candidate - filtered_id), candidate),
                    )
                    if not ask_confirm(
                        f"To be copied {filtered_id}:{class_name} ID, destination data.yaml "
                        f"{filtered_id}:{conflicting_name} is used by. nearest empty "
                        f"ID {target_id}. filtered {class_name} label ID'leri "
                        f"{target_id}? If No, the operation is cancelled.",
                        True,
                    ):
                        raise SystemExit("The free class ID suggestion was rejected; copying was cancelled.")
                copy_id_map[old_id] = target_id
                output_names[target_id] = class_name
                target_by_name[normalized] = target_id
                occupied_ids.add(target_id)
            destination_names = output_names
        else:
            existing_labels = [
                label
                for paths in split_dirs(destination, create=False).values()
                if paths["labels"].is_dir()
                for label in paths["labels"].glob("*.txt")
            ]
            if existing_labels:
                raise RuntimeError(
                    "Hedefte data.yaml none fakat existing label dosyalari exists. Class ID "
                    "anlamlari bilinmedigi for safe copying yapilamaz."
                )
            destination_names = {
                filtered_id_by_old[old_id]: names[old_id] for old_id in filtered_order
            }

    pairs = manifest(root, require_yaml=True)
    parsed_pairs = []
    for pair in pairs:
        lines, _ = parse_label(pair.label, set(names))
        present = {cid for cid, _ in lines}
        if policy == "only_effective" and (not present or not present.issubset(effective_ids)):
            continue
        parsed_pairs.append((pair, lines, present))

    chosen_by_image = {}
    requested = {}
    available = {}
    split_available = {}
    split_requested = {}
    for cid in sorted(effective_ids):
        candidates = [item for item in parsed_pairs if cid in item[2]]
        available[cid] = len(candidates)
        available_by_split = Counter(item[0].source_split for item in candidates)
        split_available[cid] = {split: available_by_split[split] for split in SPLITS}
        if not candidates:
            raise ValueError(f"{cid}:{names[cid]} for no suitable images were found.")
        raw = ask_text(
            f"{cid}:{names[cid]} for how many tane should be filtered? "
            f"(found={len(candidates)}, 0=all):",
            "0",
        )
        try:
            amount = int(raw)
        except ValueError as exc:
            raise ValueError("The filter count must be an integer of 0 or greater.") from exc
        if amount < 0 or amount > len(candidates):
            raise ValueError(
                f"{cid}:{names[cid]} for count must be between 0 and {len(candidates)} inclusive."
            )
        requested[cid] = amount
        if amount == 0:
            targets = dict(split_available[cid])
        else:
            raw_targets = {
                split: amount * split_available[cid][split] / len(candidates)
                for split in SPLITS
            }
            targets = {split: math.floor(raw_targets[split]) for split in SPLITS}
            remaining = amount - sum(targets.values())
            remainder_order = sorted(
                SPLITS,
                key=lambda split: (
                    raw_targets[split] - targets[split],
                    split_available[cid][split],
                    -SPLITS.index(split),
                ),
                reverse=True,
            )
            for split in remainder_order[:remaining]:
                targets[split] += 1
        split_requested[cid] = targets

        for split_index, split in enumerate(SPLITS):
            split_candidates = [item for item in candidates if item[0].source_split == split]
            random.Random(SEED + cid * 10 + split_index).shuffle(split_candidates)
            for pair, lines, present in split_candidates[:targets[split]]:
                chosen_by_image[pair.image.resolve()] = (pair, lines, present)

    chosen = list(chosen_by_image.values())
    chosen.sort(key=lambda item: (item[0].source_split, item[0].image.name.casefold()))
    if not chosen:
        raise ValueError("Secime uyan image-label pair not found.")
    mixed_pairs = sum(bool(present - effective_ids) for _, _, present in chosen)

    print("\nClass data operation plan:")
    print("  source:", root)
    print("  operation:", "copy to another dataset" if action == "copy" else "delete image-label pairs")
    if destination is not None:
        print("  destination:", destination)
    print("  Scope:", ", ".join(f"{cid}:{names[cid]}" for cid in sorted(effective_ids)))
    if action == "copy":
        print("  Target data.yaml class mappings:")
        for old_id in filtered_order:
            filtered_id = filtered_id_by_old[old_id]
            target_id = copy_id_map[old_id]
            print(f"    {filtered_id}:{names[old_id]} -> {target_id}:{destination_names[target_id]}")
    for cid in sorted(effective_ids):
        amount_text = "all" if requested[cid] == 0 else str(requested[cid])
        print(f"  {cid}:{names[cid]} -> requested={amount_text}, suitable={available[cid]}")
        print(
            "    existing: "
            + ", ".join(f"{split}={split_available[cid][split]}" for split in SPLITS)
        )
        print(
            "    to be selected: "
            + ", ".join(f"{split}={split_requested[cid][split]}" for split in SPLITS)
        )
    print(f"  Unique pairs to process={len(chosen)}, mixed-class images={mixed_pairs}")
    if len(effective_ids) > 1:
        print(
            "  note: If an image contains multiple target classes, it may contribute to each class selection "
            "but it is copied/deleted only before."
        )
    if action == "copy" and mixed_pairs:
        print(
            "  WARNING: Out-of-scope objects remain visible in the image, but their boxes are not written to the new "
            "label; they may be treated as background during training."
        )
    if action == "delete" and mixed_pairs:
        print(
            "  WARNING: Some selected candidates also contain out-of-scope classes. The delete operation "
            "removes these images and label files completely."
        )
    if not ask_confirm("Apply this plan?", False):
        print("No changes were made.")
        return

    if action == "delete":
        backup = backup_metadata(root, "class_pair_delete")
        if backup:
            print("Label/YAML backup:", backup)
        total = len(chosen)
        show_progress("deletion", 0, total, f"0/{total} pair")
        for index, (pair, _, _) in enumerate(chosen, 1):
            pair.image.unlink()
            pair.label.unlink()
            if index == total or index % max(1, total // 100) == 0:
                show_progress("deletion", index, total, f"{index}/{total} pair", finish=index == total)
        validate_dataset(root, False, True)
        print(f"Deletion completed: {total} image-label pairs were removed.")
        return

    if destination.exists() and destination_yaml.is_file():
        backup = backup_metadata(destination, "filter_copy")
        if backup:
            print("Target label/YAML backup:", backup)
    dirs = split_dirs(destination, create=True)
    total = len(chosen)
    show_progress("Copy class data", 0, total, f"0/{total} pair")
    copied_boxes = 0
    for index, (pair, lines, _) in enumerate(chosen, 1):
        target = dirs[pair.source_split]
        stem, _ = unique_stem(pair.image.stem, root.name, target["images"], target["labels"])
        filtered = [(cid, coords) for cid, coords in lines if cid in effective_ids]
        shutil.copy2(pair.image, target["images"] / f"{stem}{pair.image.suffix}")
        (target["labels"] / f"{stem}.txt").write_text(
            rewrite_label(filtered, copy_id_map), encoding="utf-8"
        )
        copied_boxes += len(filtered)
        if index == total or index % max(1, total // 100) == 0:
            show_progress(
                "Copy class data", index, total, f"{index}/{total} pair",
                finish=index == total,
            )
    write_yaml(destination_yaml, destination_data, destination_names)
    validate_dataset(destination, False, True)
    print(f"Copy completed: pair={total}, box={copied_boxes}, destination={destination}")


def _impl_filter_classes():
    filter_action = ask_select(
        "Class operation:",
        [questionary.Choice("(1) Keep or remove classes", "filter"),
         questionary.Choice("(2) Merge multiple classes into one class", "merge_classes"),
         questionary.Choice("(3) Convert all label class IDs to a single value", "force_id")],
    )
    root = choose_one_dataset("Dataset whose classes will be filtered:")
    repair_missing_pairs(root)
    data, names, yaml_path = load_yaml(root, required=filter_action != "force_id")
    if filter_action == "merge_classes":
        merge_classes_within_dataset(root, data, names, yaml_path)
        return
    if filter_action == "force_id":
        force_all_labels_to_class_id(root, data, names, yaml_path)
        return
    extract_or_delete_class_pairs(root, data, names)


def filter_classes():
    try:
        return _impl_filter_classes()
    except BackMenu:
        return None


def build_name_map(source_names, destination_names):
    target_by_name = {normalized_name(v): i for i, v in destination_names.items()}
    output, id_map = dict(destination_names), {}
    next_id = max(output, default=-1) + 1
    for old_id in sorted(source_names):
        key = normalized_name(source_names[old_id])
        if key not in target_by_name:
            target_by_name[key] = next_id
            output[next_id] = source_names[old_id]
            next_id += 1
        id_map[old_id] = target_by_name[key]
    return id_map, output


def validate_new_dataset_name(name):
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("New folder name cannot be empty or contain a directory path.")
    if any(char in name for char in '<>:"/\\|?*') or name.endswith((" ", ".")):
        raise ValueError("New folder name contains characters invalid on Windows/Linux.")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                *(f"lpt{i}" for i in range(1, 10))}
    if name.casefold() in reserved:
        raise ValueError("This folder name is reserved by Windows.")
    return name


def _impl_merge_datasets():
    sources = choose_ordered_datasets(
        "Select the SOURCE folders to copy into the main dataset:"
    )
    if not sources:
        raise ValueError("At least one source dataset must be selected.")

    source_info = []
    for source in sources:
        repair_missing_pairs(source)
        _, source_names, _ = load_yaml(source)
        source_info.append((source, source_names, manifest(source, True)))

    create_new = ask_confirm("Create a new destination/main dataset folder?", False)
    destination_is_new = False
    rebuild_destination_yaml = False
    if create_new:
        destination_parent = choose_directory(
            "Select the parent directory where the new destination folder will be created:", BASE_DIRECTORY
        )
        destination_name = validate_new_dataset_name(ask_text("New destination folder name:"))
        destination = destination_parent / destination_name
        if destination.exists():
            raise FileExistsError(f"A file/folder with this name already exists: {destination}")
        destination_data, destination_names = {}, {}
        destination_yaml = destination / DATA_YAML_NAME
        destination_is_new = True
        rebuild_destination_yaml = True
    else:
        destination = choose_destination_folder("Existing destination/main dataset:", set(sources))
        required_paths = [
            destination / split / kind
            for split in SPLITS for kind in ("images", "labels")
        ]
        if all(path.is_dir() for path in required_paths):
            if ask_confirm(
                "The destination already contains train/val/test images/labels folders. "
                "Recreate the same folder structure? Existing files will not be deleted.",
                False,
            ):
                split_dirs(destination, create=True)
        else:
            split_dirs(destination, create=True)
            print("Missing train/val/test images/labels folders were created in the destination.")
        repair_missing_pairs(destination)
        yaml_exists = (destination / DATA_YAML_NAME).is_file()
        destination_data, destination_names, destination_yaml = load_yaml(
            destination, required=False
        )
        if yaml_exists:
            rebuild_destination_yaml = ask_confirm(
                "The destination already contains data.yaml. according to the selected sources' data.yaml class order "
                "Recreate it according to the selected sources' data.yaml class order? Existing label IDs will also be remapped safely.",
                False,
            )
        else:
            existing_pairs = manifest(destination, require_yaml=False)
            if existing_pairs:
                raise RuntimeError(
                    "Hedefte data.yaml none fakat existing image/label exists. Old ID anlamlari "
                    "bilinemedigi for otomatik data.yaml olusturmak safe degil."
                )
            rebuild_destination_yaml = True
            print("The destination has no data.yaml; it will be created from the selected source data.yaml files.")

    mode_choices = [questionary.Choice(
        "Match by data.yaml names; append new classes at the end (recommended)", "names")]
    if not destination_is_new and not rebuild_destination_yaml:
        mode_choices.extend([
            questionary.Choice(
                "Map all non-empty boxes from each source to one selected target ID", "single"),
            questionary.Choice(
                "Copy IDs without changing them (only when meanings match exactly)", "raw"),
        ])
    mode = ask_select(
        "Class ID mapping method:",
        mode_choices,
    )

    # In a new or rebuilt YAML, the selection order starts at 0.
    # If the existing YAML is preserved, its class order comes first and new classes are appended.
    output_names = {} if rebuild_destination_yaml else dict(destination_names)
    mappings = {}
    for source, source_names, _ in source_info:
        if mode == "names":
            id_map, output_names = build_name_map(source_names, output_names)
        elif mode == "single":
            print(f"\nTarget classes ({source.name} for):")
            for cid in sorted(output_names):
                print(f"  {cid}: {output_names[cid]}")
            new_id = int(ask_text(f"{source.name} icindeki all kutularin destination class ID:"))
            if new_id not in output_names:
                raise ValueError(f"ID does not exist in the destination data.yaml: {new_id}")
            id_map = {old: new_id for old in source_names}
        else:
            if set(source_names) - set(output_names):
                raise ValueError(f"{source.name}: the destination has missing IDs; RAW mapping is unsafe.")
            bad = [i for i in source_names if normalized_name(source_names[i]) !=
                   normalized_name(output_names[i])]
            if bad:
                raise ValueError(f"{source.name}: the same ID has a different class meaning: {bad}")
            id_map = {i: i for i in source_names}
        mappings[source] = id_map

    destination_id_map = None
    destination_pairs_before_merge = []
    if rebuild_destination_yaml and not destination_is_new and destination_names:
        destination_pairs_before_merge = manifest(destination, require_yaml=True)
        target_by_name = {normalized_name(name): cid for cid, name in output_names.items()}
        used_old_ids = set()
        for pair in destination_pairs_before_merge:
            used_old_ids.update(pair.class_ids)
        destination_id_map = {}
        missing_used_classes = []
        for old_id in sorted(used_old_ids):
            key = normalized_name(destination_names[old_id])
            if key not in target_by_name:
                missing_used_classes.append(f"{old_id}:{destination_names[old_id]}")
            else:
                destination_id_map[old_id] = target_by_name[key]
        if missing_used_classes:
            raise RuntimeError(
                "The destination data.yaml cannot be rebuilt. The following classes are used in existing labels but "
                "are not present in the selected source data.yaml files: "
                + ", ".join(missing_used_classes)
            )

    print("\nClass mappings:")
    total_pairs = 0
    for source, source_names, pairs in source_info:
        print(f"  [{source.name}] image-label pair={len(pairs)}")
        total_pairs += len(pairs)
        for old in sorted(mappings[source]):
            new = mappings[source][old]
            print(f"    {old}:{source_names[old]} -> {new}:{output_names[new]}")
    print("\nMain data.yaml class order to be created:")
    for cid in sorted(output_names):
        print(f"  {cid}: {output_names[cid]}")
    print(f"\nHedef: {destination}\nTotal pairs to copy: {total_pairs}")
    if not ask_confirm("Start merging?", False):
        return

    if destination_is_new:
        split_dirs(destination, create=True)
    else:
        backup = backup_metadata(destination, "merge")
        if backup:
            print("Target label/YAML backup:", backup)
    remapped_existing = 0
    if destination_id_map is not None:
        for pair in destination_pairs_before_merge:
            lines, _ = parse_label(pair.label, set(destination_names))
            new_text = rewrite_label(lines, destination_id_map)
            old_text = pair.label.read_text(encoding="utf-8-sig")
            if new_text != old_text:
                pair.label.write_text(new_text, encoding="utf-8")
                remapped_existing += 1
        print(f"Existing destination labels remapped for the new data.yaml order: {remapped_existing}")
    dirs = split_dirs(destination, create=True)
    renamed = negatives = 0
    copied = 0
    copy_progress_step = max(1, total_pairs // 100)
    show_progress("Copying", 0, total_pairs, f"0/{total_pairs} pair")
    for source, source_names, pairs in source_info:
        id_map = mappings[source]
        for pair in pairs:
            target = dirs[pair.source_split]
            stem, changed = unique_stem(pair.image.stem, source.name,
                                        target["images"], target["labels"])
            lines, _ = parse_label(pair.label, set(source_names))
            shutil.copy2(pair.image, target["images"] / f"{stem}{pair.image.suffix}")
            text = rewrite_label(lines, id_map)
            destination_label = target["labels"] / f"{stem}.txt"
            original_text = pair.label.read_text(encoding="utf-8-sig")
            if text == original_text:
                # If the ID is already correct, copy the label without modifying its contents.
                shutil.copy2(pair.label, destination_label)
            else:
                destination_label.write_text(text, encoding="utf-8")
            copied += 1
            renamed += changed
            negatives += not text.strip()
            if copied % copy_progress_step == 0 or copied == total_pairs:
                show_progress(
                    "Copying", copied, total_pairs,
                    f"{copied}/{total_pairs} pair",
                    finish=copied == total_pairs,
                )
    print("data.yaml is being written and the merged dataset is being validated...")
    write_yaml(destination_yaml, destination_data, output_names)
    validate_dataset(destination, False, True)
    print(f"Finished: copied={copied}, renamed={renamed}, negative={negatives}")


def merge_datasets():
    try:
        return _impl_merge_datasets()
    except BackMenu:
        return None


def largest_remainder_targets(total, ratios):
    raw = [total * x / 100 for x in ratios]
    result = [math.floor(x) for x in raw]
    for i in sorted(range(3), key=lambda j: raw[j] - result[j], reverse=True)[:total-sum(result)]:
        result[i] += 1
    return dict(zip(SPLITS, result))


def multilabel_assignment(pairs, ratios, multi_class, seed):
    rng = random.Random(seed)
    size_targets = largest_remainder_targets(len(pairs), ratios)
    totals = Counter(cid for pair in pairs for cid in pair.class_ids)
    class_targets = {cid: largest_remainder_targets(n, ratios) for cid, n in totals.items()}
    ordered = pairs[:]
    rng.shuffle(ordered)
    if multi_class:
        ordered.sort(key=lambda p: (
            min((totals[c] for c in p.class_ids), default=10**12), -len(p.class_ids)))
    assigned = {s: [] for s in SPLITS}
    current = {s: Counter() for s in SPLITS}
    for pair in ordered:
        candidates = [s for s in SPLITS if len(assigned[s]) < size_targets[s]] or list(SPLITS)
        def score(split):
            size_need = (size_targets[split] - len(assigned[split])) / max(1, size_targets[split])
            label_need = 0 if not multi_class else sum(
                max(0, class_targets[c][split] - current[split][c]) /
                max(1, class_targets[c][split]) for c in pair.class_ids)
            return label_need, size_need, rng.random()
        chosen = max(candidates, key=score)
        assigned[chosen].append(pair)
        current[chosen].update(pair.class_ids)
    return assigned, size_targets, class_targets


def ask_ratios():
    ratios = tuple(float(ask_text(f"{name.capitalize()} yuzdesi:", str(int(default))))
                   for name, default in zip(SPLITS, DEFAULT_RATIOS))
    if any(x < 0 or x > 100 for x in ratios) or not math.isclose(sum(ratios), 100, abs_tol=1e-7):
        raise ValueError("ratios 0-100 arasinda and sum exactly %100 must be.")
    return ratios


def _impl_redistribute_datasets():
    declared = int(ask_text("How many class/dataset folders do you have?", "1"))
    if declared < 1:
        raise ValueError("The number of folders must be at least 1.")
    single = choose_many_datasets(
        "Select dataset folders containing a single class:", allow_empty=True
    )
    multi = choose_many_datasets(
        "Select dataset folders containing multiple classes:", exclude=set(single), allow_empty=True
    )
    selected = single + multi
    if len(selected) != declared:
        raise ValueError(f"{declared} declared, {len(selected)} folders selected.")
    ratios = ask_ratios()
    plans = []
    for root in selected:
        repair_missing_pairs(root)
        pairs = manifest(root, root in multi)
        split_dirs(root, create=True)
        assigned, targets, class_targets = multilabel_assignment(pairs, ratios, root in multi, SEED)
        print(f"\n{root.name}: total={len(pairs)}, destination={targets}")
        if root in multi:
            _, names, _ = load_yaml(root)
            for cid in sorted(class_targets):
                print(f"  {cid}:{names[cid]} image target -> {class_targets[cid]}")
        moves = sum(p.source_split != s for s, items in assigned.items() for p in items)
        print("  to be moved pair:", moves)
        plans.append((root, assigned, root in multi))
    if not ask_confirm("Apply the redistribution plans?", False):
        return
    for root, assigned, is_multi in plans:
        backup = backup_metadata(root, "split")
        if backup:
            print(f"{root.name} label/YAML backup: {backup}")
        temp = root / f"_split_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        temp.mkdir()
        staged = []
        total = sum(len(items) for items in assigned.values())
        show_progress("Redistribution - temporary move", 0, total, f"0/{total} pair")
        for split, items in assigned.items():
            for pair in items:
                index = len(staged)
                ti, tl = temp / f"{index:09d}{pair.image.suffix}", temp / f"{index:09d}.txt"
                shutil.move(str(pair.image), ti); shutil.move(str(pair.label), tl)
                staged.append((split, pair.image.stem, pair.image.suffix, ti, tl))
                done = len(staged)
                if done == total or done % max(1, total // 100) == 0:
                    show_progress(
                        "Redistribution - temporary move", done, total,
                        f"{done}/{total} pair", finish=done == total,
                    )
        dirs = split_dirs(root, create=True)
        show_progress("Redistribution - placement", 0, total, f"0/{total} pair")
        for index, (split, old_stem, suffix, ti, tl) in enumerate(staged, 1):
            target = dirs[split]
            stem, _ = unique_stem(old_stem, root.name, target["images"], target["labels"])
            shutil.move(str(ti), target["images"] / f"{stem}{suffix}")
            shutil.move(str(tl), target["labels"] / f"{stem}.txt")
            if index == total or index % max(1, total // 100) == 0:
                show_progress(
                    "Redistribution - placement", index, total,
                    f"{index}/{total} pair", finish=index == total,
                )
        temp.rmdir()
        validate_dataset(root, False, is_multi)
    print("Ratio-based redistribution completed.")


def redistribute_datasets():
    try:
        return _impl_redistribute_datasets()
    except BackMenu:
        return None


def conversion_candidates():
    """List folders using the old split/images layout for conversion."""
    result = []
    for root in sorted(BASE_DIRECTORY.iterdir(), key=lambda p: p.name.casefold()):
        if not root.is_dir():
            continue
        train_ok = (root / "train" / "images").is_dir() and (root / "train" / "labels").is_dir()
        test_ok = (root / "test" / "images").is_dir() and (root / "test" / "labels").is_dir()
        val_ok = any(
            (root / name / "images").is_dir() and (root / name / "labels").is_dir()
            for name in ("val", "valid")
        )
        if train_ok and test_ok and val_ok:
            result.append(root)
    return result


def inspect_old_layout(root):
    """Check all image-label pairs and labels before conversion."""
    if (root / "val").exists() and (root / "valid").exists():
        raise RuntimeError("Both val and valid folders exist. Choose one and use a single name.")
    val_source = "val" if (root / "val").is_dir() else "valid"
    aliases = {"train": "train", "val": val_source, "test": "test"}
    _, names, _ = load_yaml(root, required=True)
    known = set(names)
    summary = {}
    for final_split, source_split in aliases.items():
        images_dir = root / source_split / "images"
        labels_dir = root / source_split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {images_dir} or {labels_dir}")
        images = image_files(images_dir)
        labels = sorted(labels_dir.glob("*.txt"))
        image_stems = [p.stem for p in images]
        if len(image_stems) != len(set(image_stems)):
            raise RuntimeError(f"same stem ada sahip more than one more image exists: {images_dir}")
        label_stems = {p.stem for p in labels}
        missing_labels = set(image_stems) - label_stems
        missing_images = label_stems - set(image_stems)
        if missing_labels or missing_images:
            raise RuntimeError(
                f"{source_split} mapping error; images without labels={sorted(missing_labels)[:10]}, "
                f"labels without images={sorted(missing_images)[:10]}"
            )
        boxes = negatives = 0
        for label in labels:
            lines, _ = parse_label(label, known)
            boxes += len(lines)
            negatives += int(not lines)
        summary[final_split] = (len(images), len(labels), boxes, negatives)
    return aliases, summary


def validate_final_layout(root):
    """Validate the ZIP-ready images/split + labels/split layout."""
    _, names, _ = load_yaml(root, required=True)
    known = set(names)
    print(f"\nFinal layout validation: {root}")
    for split in ("train", "val", "test"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = image_files(images_dir)
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        if image_stems != label_stems:
            raise RuntimeError(
                f"{split}: image-label do not match; without labels={sorted(image_stems-label_stems)[:10]}, "
                f"without images={sorted(label_stems-image_stems)[:10]}"
            )
        boxes = negatives = 0
        for label in labels:
            lines, _ = parse_label(label, known)
            boxes += len(lines)
            negatives += int(not lines)
        print(
            f"  {split:5s}: image={len(images)}, label={len(labels)}, "
            f"box={boxes}, negative={negatives} [OK]"
        )


def _impl_convert_to_zip_layout():
    root = choose_directory(
        "MAIN dataset to convert to the ZIP-ready images/split + labels/split layout:",
        BASE_DIRECTORY,
    ).resolve()
    if root not in conversion_candidates_for_root(root):
        raise ValueError(
            "selected folder train/{images,labels}, val or valid/{images,labels}, "
            "test/{images,labels} is not in the expected working layout."
        )
    if (root / "images").exists() or (root / "labels").exists():
        raise RuntimeError(
            "The destination already contains an images or labels folder. Operation stopped to avoid overwriting existing data."
        )
    aliases, summary = inspect_old_layout(root)
    print("\nConversion plan:")
    for split in ("train", "val", "test"):
        ni, nl, nb, ne = summary[split]
        print(f"  {aliases[split]} -> {split}: image={ni}, label={nl}, box={nb}, negative={ne}")
    print("\nOld: train/images + train/labels")
    print("New: images/train + labels/train")
    print("WARNING: This is the final packaging step; do not run operations 1-3 afterwards.")
    if not ask_confirm("Convert the main dataset structure now?", False):
        print("No changes were made.")
        return

    data, names, yaml_path = load_yaml(root, required=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    yaml_backup = root.parent / f"{root.name}_layout_yaml_backup_{stamp}.zip"
    with ZipFile(yaml_backup, "w", ZIP_DEFLATED) as archive:
        archive.write(yaml_path, yaml_path.relative_to(root))
    print("data.yaml backup:", yaml_backup)

    temp = root / f"_layout_conversion_temp_{stamp}"
    if temp.exists():
        raise FileExistsError(f"Gecici folder zaten exists: {temp}")
    (temp / "images").mkdir(parents=True)
    (temp / "labels").mkdir(parents=True)
    try:
        for final_split, source_split in aliases.items():
            shutil.move(str(root / source_split / "images"), temp / "images" / final_split)
            shutil.move(str(root / source_split / "labels"), temp / "labels" / final_split)
        for source_split in set(aliases.values()):
            (root / source_split).rmdir()
        shutil.move(str(temp / "images"), root / "images")
        shutil.move(str(temp / "labels"), root / "labels")
        temp.rmdir()
    except Exception:
        print(f"Conversion stopped partway through. Recovery files are here: {temp}")
        raise

    output = dict(data)
    output["train"] = "images/train"
    output["val"] = "images/val"
    output["test"] = "images/test"
    output.pop("path", None)
    write_yaml(yaml_path, output, names)
    # write_yaml writes the working layout; final package paths are fixed one last time.
    output = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    output["train"], output["val"], output["test"] = (
        "images/train", "images/val", "images/test"
    )
    yaml_path.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")
    validate_final_layout(root)
    print("Main dataset converted to the ZIP-ready layout.")


def convert_to_zip_layout():
    try:
        return _impl_convert_to_zip_layout()
    except BackMenu:
        return None


def final_layout_candidates():
    """Items using the images/{train,val,test} + labels/{train,val,test} layout."""
    result = []
    for root in sorted(BASE_DIRECTORY.iterdir(), key=lambda p: p.name.casefold()):
        if not root.is_dir() or not (root / DATA_YAML_NAME).is_file():
            continue
        if all(
            (root / kind / split).is_dir()
            for kind in ("images", "labels")
            for split in ("train", "val", "test")
        ):
            result.append(root)
    return result


def conversion_candidates_for_root(root):
    train_ok = (root / "train" / "images").is_dir() and (root / "train" / "labels").is_dir()
    test_ok = (root / "test" / "images").is_dir() and (root / "test" / "labels").is_dir()
    val_ok = any(
        (root / name / "images").is_dir() and (root / name / "labels").is_dir()
        for name in ("val", "valid")
    )
    return [root] if root.is_dir() and train_ok and val_ok and test_ok else []


def _impl_create_dataset_zip():
    root = choose_directory("Main dataset to ZIP:", BASE_DIRECTORY).resolve()
    if not (
        (root / DATA_YAML_NAME).is_file()
        and all(
            (root / kind / split).is_dir()
            for kind in ("images", "labels") for split in SPLITS
        )
    ):
        raise ValueError(
            "The selected folder is not suitable for ZIP images/{train,val,test} + "
            "labels/{train,val,test} layout."
        )
    validate_final_layout(root)
    zip_directory = choose_directory("Directory where the ZIP file will be saved:", root.parent).resolve()
    output = zip_directory / f"{root.name}.zip"
    temporary = zip_directory / f".{root.name}.zip.part"
    if output.exists() and not ask_confirm(
        f"{output.name} already exists. Replace it with the newly validated ZIP?", False
    ):
        print("ZIP creation cancelled.")
        return
    if temporary.exists():
        temporary.unlink()

    files = [root / DATA_YAML_NAME]
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            files.extend(sorted(p for p in (root / kind / split).rglob("*") if p.is_file()))
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"ZIP file: {output}")
    print(f"File count: {len(files):,}; raw size: {total_bytes / (1024**3):.2f} GB")
    if not ask_confirm("Start ZIP creation?", True):
        print("ZIP creation cancelled.")
        return

    try:
        # ZIP64 supports large datasets. The low compression level avoids unnecessary CPU usage on JPG/PNG
        # while still compressing txt/yaml files.
        with ZipFile(
            temporary, "w", ZIP_DEFLATED, allowZip64=True, compresslevel=1
        ) as archive:
            # Include empty split folders in the archive as well.
            for kind in ("images", "labels"):
                for split in ("train", "val", "test"):
                    archive.writestr(f"{kind}/{split}/", "")
            processed_bytes = 0
            last_percentage = -1
            show_progress(
                "ZIP creation", 0, max(1, total_bytes),
                f"0/{len(files)} file | 0.00/{total_bytes / (1024**3):.2f} GB",
            )
            for index, file in enumerate(files, 1):
                archive.write(file, file.relative_to(root).as_posix())
                processed_bytes += file.stat().st_size
                percentage = int(
                    processed_bytes * 100 / total_bytes
                ) if total_bytes else int(index * 100 / max(1, len(files)))
                if percentage != last_percentage or index == len(files):
                    last_percentage = percentage
                    progress_current = processed_bytes if total_bytes else index
                    progress_total = total_bytes if total_bytes else len(files)
                    show_progress(
                        "ZIP creation", progress_current, max(1, progress_total),
                        f"{index}/{len(files)} file | "
                        f"{processed_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GB",
                        finish=index == len(files),
                    )
        # The temporary ZIP is fully closed first; it is then renamed to the final name in one step.
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            print(f"Partial ZIP was not moved into place of the final file: {temporary}")
        raise

    print("Checking ZIP integrity...")
    with ZipFile(output, "r") as archive:
        bad = archive.testzip()
        required = {"data.yaml", "images/train/", "images/val/", "images/test/",
                    "labels/train/", "labels/val/", "labels/test/"}
        missing = required - set(archive.namelist())
    if bad or missing:
        raise RuntimeError(f"ZIP validation failed; corrupted={bad}, missing={sorted(missing)}")
    print(f"ZIP created and validated: {output}")


def create_dataset_zip():
    try:
        return _impl_create_dataset_zip()
    except BackMenu:
        return None


def validate_dataset(root, interactive=True, require_yaml=False):
    if interactive:
        repair_missing_pairs(root)
    _, names, _ = load_yaml(root, required=require_yaml)
    known = set(names) if names else None
    errors, total_images, total_labels, total_boxes, total_empty = [], 0, 0, 0, 0
    class_boxes, summaries = Counter(), {}
    all_labels = [
        label
        for paths in split_dirs(root, create=True).values()
        for label in sorted(paths["labels"].glob("*.txt"))
    ]
    processed_labels = 0
    show_progress("Dataset validation", 0, len(all_labels), f"0/{len(all_labels)} label")
    for split, paths in split_dirs(root, create=True).items():
        images, labels = image_files(paths["images"]), sorted(paths["labels"].glob("*.txt"))
        i_stems, l_stems = {p.stem for p in images}, {p.stem for p in labels}
        errors += [f"{root.name}/{split}: {s} image ait label none" for s in i_stems-l_stems]
        errors += [f"{root.name}/{split}: {s}.txt label has no corresponding image" for s in l_stems-i_stems]
        boxes = empty = 0
        for label in labels:
            try:
                lines, counts = parse_label(label, known)
                boxes += len(lines); empty += not lines; class_boxes.update(counts)
            except ValueError as exc:
                errors.append(str(exc))
            processed_labels += 1
            if (processed_labels == len(all_labels) or
                    processed_labels % max(1, len(all_labels) // 100) == 0):
                show_progress(
                    "Dataset validation", processed_labels, len(all_labels),
                    f"{processed_labels}/{len(all_labels)} label",
                    finish=processed_labels == len(all_labels),
                )
        summaries[split] = len(images), len(labels), boxes, empty
        total_images += len(images); total_labels += len(labels)
        total_boxes += boxes; total_empty += empty
    print(f"\nValidation: {root}")
    for split, (ni, nl, nb, ne) in summaries.items():
        print(f"  {split:5s}: image={ni}, label={nl}, box={nb}, negative={ne} "
              f"[{'OK' if ni == nl else 'ERROR'}]")
    if names:
        for cid in sorted(names):
            print(f"  class {cid}:{names[cid]} box={class_boxes[cid]}")
    print(f"  total: image={total_images}, label={total_labels}, "
          f"box={total_boxes}, negative={total_empty}")
    if errors or total_images != total_labels:
        raise RuntimeError(f"Validation basarisiz ({len(errors)} error):\n" + "\n".join(errors[:30]))
    print("  RESULT: image-label pairs and YOLO labels are valid.")
    return summaries


def _impl_validation_menu():
    roots = choose_many_datasets("Select datasets to validate:")
    if not roots:
        raise ValueError("At least one dataset must be selected.")
    for root in roots:
        validate_dataset(root, True, False)


def validation_menu():
    try:
        return _impl_validation_menu()
    except BackMenu:
        return None


def _impl_create_empty_labels_for_images():
    images_dir = choose_directory(
        "images folder containing the photos for which empty labels will be created:",
        BASE_DIRECTORY,
    ).resolve()
    images = image_files(images_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in the selected directory: {images_dir}")
    stems = [image.stem for image in images]
    duplicate_stems = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicate_stems:
        raise RuntimeError(
            "Multiple images have the same stem; one label name cannot belong to two images: "
            + ", ".join(duplicate_stems[:20])
        )

    labels_dir = images_dir.parent / "labels"
    existing_items = list(labels_dir.iterdir()) if labels_dir.is_dir() else []
    existing_labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    matching_existing = [labels_dir / f"{image.stem}.txt" for image in images]
    overwrite_count = sum(label.is_file() for label in matching_existing)

    print("\nEmpty label creation plan:")
    print("  Images:", images_dir)
    print("  Labels:", labels_dir)
    print(f"  image={len(images)}, matching existing labels={overwrite_count}")
    if existing_items:
        if not ask_confirm(
            "The Labels folder is not empty. Label files with the same names as the images "
            "will be written with empty content. Do you want me to overwrite them?",
            False,
        ):
            print("Create empty labels cancel edildi.")
            return
    elif not ask_confirm(
        "Create an empty .txt label file with the same name for every image?",
        False,
    ):
        print("Create empty labels cancel edildi.")
        return

    if existing_labels:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = images_dir.parent / f"labels_empty_backup_{stamp}.zip"
        with ZipFile(backup, "w", ZIP_DEFLATED) as archive:
            show_progress("Existing label backup", 0, len(existing_labels), f"0/{len(existing_labels)}")
            for index, label in enumerate(existing_labels, 1):
                archive.write(label, f"labels/{label.name}")
                if index == len(existing_labels) or index % max(1, len(existing_labels) // 100) == 0:
                    show_progress(
                        "Existing label backup", index, len(existing_labels),
                        f"{index}/{len(existing_labels)}", finish=index == len(existing_labels),
                    )
        print("Existing label backup:", backup)

    labels_dir.mkdir(parents=True, exist_ok=True)
    show_progress("Create empty labels", 0, len(images), f"0/{len(images)} label")
    for index, image in enumerate(images, 1):
        (labels_dir / f"{image.stem}.txt").write_text("", encoding="utf-8")
        if index == len(images) or index % max(1, len(images) // 100) == 0:
            show_progress(
                "Create empty labels", index, len(images), f"{index}/{len(images)} label",
                finish=index == len(images),
            )
    invalid = [label for label in matching_existing if not label.is_file() or label.stat().st_size != 0]
    if invalid:
        raise RuntimeError("Empty label validation failed: " + ", ".join(map(str, invalid[:20])))
    print(f"Completed: {len(images)} empty labels prepared for images: {labels_dir}")


def create_empty_labels_for_images():
    try:
        return _impl_create_empty_labels_for_images()
    except BackMenu:
        return None


def main():
    print(f"\nYOLO Dataset Management Tool\nWorking directory: {BASE_DIRECTORY}\n")

    # Navigation hierarchy:
    # Windows/Linux menu
    #   -> Linux distribution menu (only when Linux is selected)
    #       -> Main menu
    #
    # Therefore:
    # Linux selected + distro selected + Main menu -> Back = distro menu
    # distro menu -> Back = Windows/Linux menu
    # Windows selected + Main menu -> Back = Windows/Linux menu

    while True:
        # Level 1: Windows / Linux selection
        try:
            configure_platform()
        except BackMenu:
            return

        while True:
            # Level 2: Linux distribution selection
            #
            # configure_platform() normally already selected the distro.
            # If Linux was selected, keep that distro as the parent menu state.
            #
            # To make Back from the main menu return to the distro menu,
            # we recreate the distro selection here only when the user backs
            # out of the main menu.
            try:
                ensure_structure_confirmation()
            except BackMenu:
                # Confirmation cancellation returns to the distro/OS level.
                if SELECTED_OS == "linux":
                    try:
                        SELECTED_LINUX_DISTRO = ask_select(
                            "Your Linux distribution:",
                            [questionary.Choice("(1) Arch / Arch-based", "arch"),
                             questionary.Choice("(2) Debian / Ubuntu-based", "debian"),
                             questionary.Choice("(3) Fedora", "fedora")],
                        )
                        continue
                    except BackMenu:
                        break
                else:
                    break

            try:
                while True:
                    action = ask_select(
                        "Select an operation:",
                        [questionary.Choice("(1) Create empty/negative labels for images", "empty_labels"),
                         questionary.Choice("(2) Filter/reduce classes", "filter"),
                         questionary.Choice("(3) Merge datasets into the main dataset", "merge"),
                         questionary.Choice("(4) Redistribute using train/val/test ratios", "split"),
                         questionary.Choice("(5) Convert the main dataset to images/split + labels/split layout", "convert"),
                         questionary.Choice("(6) Validate datasets only", "validate"),
                         questionary.Choice("(7) Create a ZIP file from the main dataset", "zip"),
                         questionary.Choice("(0) Exit", "exit")],
                    )

                    if action == "exit":
                        print("Exited.")
                        return

                    {"empty_labels": create_empty_labels_for_images,
                     "filter": filter_classes, "merge": merge_datasets,
                     "split": redistribute_datasets, "convert": convert_to_zip_layout,
                     "validate": validation_menu, "zip": create_dataset_zip}[action]()
            except BackMenu:
                # Main menu -> previous menu.
                if SELECTED_OS == "linux":
                    # Linux: go back to the distribution menu.
                    try:
                        SELECTED_LINUX_DISTRO = ask_select(
                            "Your Linux distribution:",
                            [questionary.Choice("(1) Arch / Arch-based", "arch"),
                             questionary.Choice("(2) Debian / Ubuntu-based", "debian"),
                             questionary.Choice("(3) Fedora", "fedora")],
                        )
                        continue
                    except BackMenu:
                        # Distribution menu -> Windows/Linux menu.
                        break
                else:
                    # Windows has no distribution menu.
                    break

        # Returned from Linux distribution menu or Windows main menu.
        # Loop back to the Windows/Linux selection menu.
        continue


if __name__ == "__main__":
    try:
        main()
    except BackMenu:
        print("\nThere is no previous menu to return to.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by the user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
