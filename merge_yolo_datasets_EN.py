"""Interactive YOLO dataset management tool.

Structure: dataset/{train,val,test}/{images,labels} and data.yaml
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
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
except ImportError:
    tk = None
    ttk = None
    Image = None
    ImageTk = None

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
    raise SystemExit("Installation: py -m pip install questionary PyYAML") from exc[cite: 2]

BASE_DIRECTORY = Path(__file__).resolve().parent[cite: 2]
SPLITS = ("train", "val", "test")[cite: 2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}[cite: 2]
DATA_YAML_NAME = "data.yaml"[cite: 2]
DEFAULT_RATIOS = (80.0, 10.0, 10.0)[cite: 2]
SEED = 42[cite: 2]
CREATE_BACKUP = True[cite: 2]


def ask_save_backup(purpose, description, default_name):
    """Ask whether a ZIP backup is wanted BEFORE asking for its filename."""
    should_save = ask_confirm(
        f"Do you want to create a save file for {description}?",
        True,
    )
    if not should_save:
        return False, None

    filename = ask_text(
        f"Save file name for {description} (example: {default_name}.zip):",
        f"{default_name}.zip",
    )
    if not filename:
        raise ValueError("Save file name cannot be empty.")
    if not filename.lower().endswith(".zip"):
        filename += ".zip"
    filename = Path(filename).name
    if filename in {".", ".."} or any(char in filename for char in '<>:"/\\|?*'):
        raise ValueError("Invalid save file name.")
    return True, filename


def create_selected_backup(root, purpose, description, default_name, files=None, include_yaml=True):
    """Create a user-approved ZIP backup. Returns the ZIP path or None."""
    should_save, filename = ask_save_backup(purpose, description, default_name)
    if not should_save:
        print("Save file was not created.")
        return None

    target = root.parent / filename
    if target.exists():
        if not ask_confirm(f"{target} already exists. Overwrite?", False):
            print("Save operation cancelled.")
            return None

    if files is None:
        files = []
        if include_yaml:
            yml = root / DATA_YAML_NAME
            if yml.is_file():
                files.append(yml)
        files.extend(
            label
            for paths in split_dirs(root).values()
            if paths["labels"].is_dir()
            for label in sorted(paths["labels"].glob("*.txt"))
        )

    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for file in files:
            if file.is_file():
                archive.write(file, file.relative_to(root))
    print("Save file:", target)
    return target

# Ensure "*" is shown as indicator for selected items in checkboxes.
questionary_common.INDICATOR_SELECTED = "*"[cite: 2]
questionary_common.INDICATOR_UNSELECTED = " "[cite: 2]

# Determined once at program start by configure_platform().
SELECTED_OS = None[cite: 2]
SELECTED_LINUX_DISTRO = None[cite: 2]


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
    # Prevents numbers written in choice titles from being added a second time as "1)" by questionary.
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
        instruction="(Arrow keys: navigate, Space: select/unselect, Enter: confirm)",
    ).ask()
    if value is None:
        raise KeyboardInterrupt
    if "__BACK__" in value:
        raise BackMenu
    return value


def normalized_name(value):
    return " ".join(str(value).strip().casefold().split())


def show_progress(title, current, total, detail="", finish=False):
    """Single-line percentage progress in Windows and Linux terminals."""
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
    """Gets the OS selection once to be used throughout the program."""
    global SELECTED_OS, SELECTED_LINUX_DISTRO

    while True:
        SELECTED_OS = ask_select(
            "Which operating system are you running on?",
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
            f"WARNING: Python environment appears to be {detected!r} but "
            f"you selected {SELECTED_OS!r}. OS-specific operations will run based on your selection."
        )


def split_dirs(root, create=False):
    result = {}
    for split in SPLITS:
        split_name = split
        # Support reading old Roboflow folders; newly created folders will be "val".
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
        raise ValueError("This folder cannot be selected again or as a destination in this operation.")
    if not looks_like_dataset(folder):
        raise ValueError(
            "Selected folder is not a dataset containing images or labels under train/val/test: "
            f"{folder}"
        )
    return folder


def choose_destination_folder(message, exclude=None):
    """Selects an empty or existing destination folder via terminal explorer."""
    excluded = {p.resolve() for p in (exclude or set())}
    folder = choose_directory(message, BASE_DIRECTORY).resolve()
    if folder in excluded:
        raise ValueError("Folder selected as source cannot be used as destination.")
    return folder


def choose_many_datasets(message, exclude=None, allow_empty=False):
    """Selects one or more dataset folders from different directories."""
    excluded = {p.resolve() for p in (exclude or set())}
    selected = []
    if allow_empty and not ask_confirm(f"{message} Should any folder be selected?", False):
        return selected
    while True:
        folder = choose_one_dataset(message, excluded | {p.resolve() for p in selected})
        selected.append(folder)
        print(f"Selected ({len(selected)}): {folder}")
        if not ask_confirm("Add another dataset folder to this group?", False):
            return selected


def choose_ordered_datasets(message, available=None):
    """Selects sources from different directories preserving selection order."""
    print(
        "Classes from data.yaml files of selected folders will be used in main data.yaml; "
        "folder names will not be used as class names."
    )
    print("Selection order determines main data.yaml class order.")
    return choose_many_datasets(message, allow_empty=False)


def choose_directory(message, start=None):
    """Selects a destination directory by navigating folders in terminal."""
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
            tokens.append(("class:instruction", "  (No subdirectories or no read permission)\n"))
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
    """Return images directly in a folder."""
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_split_dirs(images_root):
    """Support both images/*.jpg and images/{train,val/valid,test}/*.jpg layouts."""
    if not images_root.is_dir():
        return {}

    result = {}
    for split in ("train", "val", "test"):
        candidates = [split]
        if split == "val":
            candidates.append("valid")

        found = next(
            (images_root / name for name in candidates
             if (images_root / name).is_dir()),
            None,
        )
        result[split] = found if found is not None else images_root
    return result


def labels_root_for_images(images_root):
    """Find the labels root corresponding to an images root."""
    if not images_root.is_dir():
        return None

    # Standard sibling layout:
    # dataset/images/... + dataset/labels/...
    sibling = images_root.parent / "labels"
    if sibling.is_dir():
        return sibling

    # If the selected directory itself is inside images/, use its parent.
    if images_root.name.casefold() in {"train", "val", "valid", "test"}:
        parent_images = images_root.parent
        sibling = parent_images.parent / "labels"
        if sibling.is_dir():
            return sibling

    return None


def image_label_layout(images_root):
    """
    Resolve supported layouts.

    Supported:
      1) dataset/images + dataset/labels
      2) dataset/images/{train,val,test} + dataset/labels/{train,val,test}
      3) dataset/images/{train,valid,test} + dataset/labels/{train,valid,test}
    """
    images_root = images_root.resolve()

    split_names = {
        "train": "train",
        "val": "val",
        "test": "test",
    }

    # If the user selected images/train, images/val, images/valid or images/test,
    # normalize back to the images root.
    if images_root.name.casefold() in {"train", "val", "valid", "test"}:
        parent = images_root.parent
        if parent.name.casefold() == "images":
            images_root = parent

    labels_root = labels_root_for_images(images_root)
    if labels_root is None:
        raise FileNotFoundError(
            f"Labels folder could not be found next to the selected images directory: {images_root}"
        )

    image_dirs = image_split_dirs(images_root)
    label_dirs = {}

    nested = any(
        image_dirs[split] != images_root
        for split in ("train", "val", "test")
    )

    if nested:
        for split in ("train", "val", "test"):
            candidate_names = [split]
            if split == "val":
                candidate_names.append("valid")
            label_dir = next(
                (
                    labels_root / name
                    for name in candidate_names
                    if (labels_root / name).is_dir()
                ),
                None,
            )
            if label_dir is None:
                raise FileNotFoundError(
                    f"Labels folder could not be found for {split}: {labels_root}"
                )
            label_dirs[split] = label_dir
    else:
        for split in ("train", "val", "test"):
            label_dirs[split] = labels_root

    return images_root, labels_root, image_dirs, label_dirs


def open_path(path):
    try:
        if SELECTED_OS == "windows":
            if os.name != "nt":
                raise RuntimeError("Windows image opening operation cannot be run outside Windows.")
            os.startfile(str(path))
        elif SELECTED_OS == "linux":
            if shutil.which("xdg-open") is None:
                install_commands = {
                    "arch": "sudo pacman -S xdg-utils",
                    "debian": "sudo apt install xdg-utils",
                    "fedora": "sudo dnf install xdg-utils",
                }
                command = install_commands.get(SELECTED_LINUX_DISTRO, "install xdg-utils package")
                raise RuntimeError(f"xdg-open not found. Installation: {command}")
            subprocess.Popen(["xdg-open", str(path)])
        else:
            raise RuntimeError("Operating system selection was not made.")
    except Exception as exc:
        print(f"Failed to open image: {exc}")


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
            raise ValueError(f"Class ID {class_id} not present in data.yaml: {label}:{line_no}")
        is_box = len(parts) == 5
        is_polygon = len(coords) >= 6 and len(coords) % 2 == 0
        if not (is_box or is_polygon):
            raise ValueError(f"Invalid box/polygon format: {label}:{line_no}")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
            raise ValueError(f"Coordinate out of 0-1 range: {label}:{line_no}")
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
        raise ValueError(f"Empty or duplicate class name: {path}; duplicate={duplicate}")
    return data, mapping, path


def load_yaml_for_annotation(path):
    """
    Load data.yaml for the annotation GUI without rejecting duplicate names.

    The main dataset-management workflow keeps its stricter validation:
    duplicate class names are still rejected there. Annotation is different:
    the numeric class ID is the actual identifier, so two IDs with the same
    display name can still be selected unambiguously in the GUI.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    raw_names = data.get("names")

    if isinstance(raw_names, list):
        names = {i: str(value).strip() for i, value in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(i): str(value).strip() for i, value in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Class IDs in data.yaml are invalid: {path}"
            ) from exc
    else:
        raise ValueError(f"No valid names found in data.yaml: {path}")

    if not names:
        raise ValueError(f"No classes found in data.yaml: {path}")

    invalid_ids = [cid for cid in names if cid < 0]
    empty_names = [cid for cid, name in names.items() if not name]
    if invalid_ids:
        raise ValueError(
            f"Negative class ID found in data.yaml: {invalid_ids}"
        )
    if empty_names:
        raise ValueError(
            f"Empty class name found in data.yaml: {empty_names}"
        )

    return data, names, path


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
        'SPLITS=("train", "val", "test"). Do you confirm '
        "split/images and split/labels structure in each dataset folder?", True
    ):
        raise SystemExit("Fix the folder structure and run again.")


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
            raise RuntimeError(f"Multiple images with the same base name: {stem} -> {files}")
        for image in list(images):
            label = paths["labels"] / f"{image.stem}.txt"
            if label.is_file():
                continue
            action = ask_select(
                f"Label not found for file {image.name} ({root.name}/{split}).",
                [questionary.Choice("(1) Open image", "open"),
                 questionary.Choice("(2) Continue / skip for now", "continue"),
                 questionary.Choice("(3) Terminate program", "exit")],
            )
            if action == "exit":
                raise SystemExit("Terminated due to missing label.")
            if action == "continue":
                continue
            open_path(image)
            resolution = ask_select(
                f"What should be done for {image.name}?",
                [questionary.Choice("(1) Delete image", "delete"),
                 questionary.Choice("(2) Create empty label", "empty"),
                 questionary.Choice("(3) Do nothing and exit", "exit")],
            )
            if resolution == "delete":
                image.unlink()
            elif resolution == "empty":
                label.touch()
            else:
                raise SystemExit("Terminated by user request.")
        stems = {p.stem for p in image_files(paths["images"])}
        orphans = [p for p in paths["labels"].glob("*.txt") if p.stem not in stems]
        if orphans:
            raise RuntimeError("Label files with missing images:\n" +
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
    show_progress("Scanning dataset", 0, len(items), f"0/{len(items)} images")
    for index, (split, paths, image) in enumerate(items, 1):
            label = paths["labels"] / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Label not found: {label}")
            _, counts = parse_label(label, known)
            result.append(Pair(image, label, split, frozenset(counts), counts))
            if index == len(items) or index % max(1, len(items) // 100) == 0:
                show_progress(
                    "Scanning dataset", index, len(items), f"{index}/{len(items)} images",
                    finish=index == len(items),
                )
    return result


def backup_metadata(root, purpose):
    if not CREATE_BACKUP:
        return None

    default_name = f"{root.name}_{purpose}_backup"
    should_save, filename = ask_save_backup(
        purpose,
        f"backup of existing files in dataset {root.name}",
        default_name,
    )
    if not should_save:
        print("Save file was not created.")
        return None

    target = root.parent / filename
    if target.exists():
        if not ask_confirm(f"{target} already exists. Overwrite?", False):
            print("Save operation cancelled.")
            return None

    yml = root / DATA_YAML_NAME
    files = ([yml] if yml.is_file() else []) + [
        label
        for paths in split_dirs(root).values()
        if paths["labels"].is_dir()
        for label in sorted(paths["labels"].glob("*.txt"))
    ]

    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        show_progress("Creating save file", 0, len(files), f"0/{len(files)} files")
        for index, file in enumerate(files, 1):
            archive.write(file, file.relative_to(root))
            if index == len(files) or index % max(1, len(files) // 100) == 0:
                show_progress(
                    "Creating save file", index, len(files), f"{index}/{len(files)} files",
                    finish=index == len(files),
                )
    print("Save file:", target)
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
            raise ValueError(f"No target mapping for Class ID: {old_id}")
        output.append(" ".join([str(id_map[old_id]), *coords]))
    return "\n".join(output) + ("\n" if output else "")


def merge_classes_within_dataset(root, data, names, yaml_path):
    selected = set(ask_checkbox(
        "Select classes to merge into a single class:",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(names)],
    ))
    if len(selected) < 2:
        raise ValueError("At least two classes must be selected to merge.")
    target_id = ask_select(
        "Which target class name should the selected classes merge into?",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(selected)],
    )

    # Target class stays in its old position; other merged classes are removed.
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
    show_progress("Scanning class merge", 0, len(pairs), f"0/{len(pairs)} labels")
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
                "Scanning class merge", index, len(pairs), f"{index}/{len(pairs)} labels",
                finish=index == len(pairs),
            )

    print("\nClass merge plan:")
    for cid in sorted(selected):
        print(
            f"  {cid}:{names[cid]} -> "
            f"{retained_to_new[target_id]}:{names[target_id]} "
            f"(boxes={selected_box_counts[cid]})"
        )
    print("\nNew data.yaml class order:")
    for cid in sorted(new_names):
        print(f"  {cid}: {new_names[cid]}")
    print(
        f"Labels to change={changed_labels}, boxes to convert to target class={merged_boxes}, "
        f"empty/negative labels={empty_labels}"
    )
    if not ask_confirm("Apply class merge plan?", False):
        print("No changes made.")
        return

    backup = backup_metadata(root, "class_merge")
    if backup:
        print("Label/YAML backup:", backup)
    show_progress("Merging classes", 0, len(plans), f"0/{len(plans)} labels")
    for index, (label, text) in enumerate(plans, 1):
        label.write_text(text, encoding="utf-8")
        if index == len(plans) or index % max(1, len(plans) // 100) == 0:
            show_progress(
                "Merging classes", index, len(plans), f"{index}/{len(plans)} labels",
                finish=index == len(plans),
            )
    write_yaml(yaml_path, data, new_names)
    validate_dataset(root, interactive=False, require_yaml=True)
    print(
        f"Class merge completed: {', '.join(names[c] for c in sorted(selected))} "
        f"-> {names[target_id]}"
    )


def force_all_labels_to_class_id(root, data, names, yaml_path):
    """
    Converts class IDs of all filled label lines to a single target ID.

    Target can be selected in two ways:
      1) An existing class from data.yaml
      2) A new class ID specified by the user

    When a new ID is entered, the user is additionally asked whether
    it should also be added to data.yaml. If No, existing data.yaml is preserved.
    """

    created_new_class = False

    if names:
        target_choice = ask_select(
            "Which target class should all filled label lines be converted to?",
            [
                questionary.Choice(
                    "Select a class from data.yaml",
                    "yaml_class",
                ),
                questionary.Choice(
                    "Specify new class ID yourself",
                    "new_id",
                ),
            ],
        )
    else:
        print(
            "No selectable class found in data.yaml; "
            "you must specify the new class ID yourself."
        )
        target_choice = "new_id"

    if target_choice == "yaml_class":
        new_id = ask_select(
            "Select target class from data.yaml:",
            [
                questionary.Choice(
                    f"{cid}: {names[cid]}",
                    cid,
                )
                for cid in sorted(names)
            ],
        )
        target_name = names[new_id]

    else:
        raw_id = ask_text(
            "Enter new class ID (an integer 0 or greater):"
        )
        try:
            new_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(
                "Class ID must be an integer 0 or greater."
            ) from exc

        if new_id < 0:
            raise ValueError("Class ID cannot be negative.")

        if new_id in names:
            raise ValueError(
                f"ID {new_id} already exists in data.yaml: "
                f"{names[new_id]}. To select an existing class, "
                "use 'Select a class from data.yaml' option from the previous menu."
            )

        target_name = ask_text(
            "Enter name for the new class (used if written to data.yaml):"
        )
        if not target_name:
            raise ValueError("Class name cannot be empty.")

        # If another class with the same name exists, creating a new class name
        # is meaningless in terms of data.yaml.
        normalized_target = normalized_name(target_name)
        duplicate_name = next(
            (
                cid
                for cid, name in names.items()
                if normalized_name(name) == normalized_target
            ),
            None,
        )

        if duplicate_name is not None and duplicate_name != new_id:
            # If the same class name exists under another ID, DO NOT THROW ERROR.
            # Ask user and if approved, remove old ID from data.yaml.
            overwrite_name = ask_confirm(
                f"Name '{target_name}' is already used in data.yaml "
                f"under ID {duplicate_name}. "
                f"Allow moving to ID {new_id} and removing old ID {duplicate_name} "
                "from data.yaml?",
                False,
            )

            if not overwrite_name:
                raise ValueError(
                    f"Name '{target_name}' is already used under ID {duplicate_name}; "
                    "operation cancelled."
                )

            names = dict(names)
            del names[duplicate_name]

        created_new_class = True

    if new_id != 0:
        print(
            "WARNING: Training ID for a single-class dataset should normally be 0. "
            f"Value {new_id} can be used for preparation before merging into main dataset; "
            "do not train this intermediate dataset directly."
        )

    # The goal of this mode is to repair broken/old IDs. Therefore, old ID is not
    # required to be defined in data.yaml; line format and coordinates
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

        new_text = rewrite_label(
            lines,
            {old_id: new_id for old_id in counts},
        )
        old_text = pair.label.read_text(encoding="utf-8-sig")

        if new_text != old_text:
            plans.append((pair.label, new_text))
        else:
            already_correct += int(bool(lines))

    print("\nBulk class ID change plan:")
    for old_id in sorted(old_id_counts):
        old_name = names.get(old_id, "<not defined in data.yaml>")
        print(
            f"  {old_id}:{old_name} -> "
            f"{new_id}:{target_name} "
            f"(boxes={old_id_counts[old_id]})"
        )

    print(
        f"Total boxes={box_count}, labels to change={len(plans)}, "
        f"already correct filled labels={already_correct}, "
        f"empty/negative labels={empty_count}"
    )

    if created_new_class:
        print(
            f"\nNew class:"
            f"\n  ID   : {new_id}"
            f"\n  Name : {target_name}"
        )

        update_yaml = ask_confirm(
            "Do you want this new class ID to be recorded into data.yaml as well?",
            True,
        )
    else:
        # When an existing data.yaml class is selected, no extra class addition
        # is needed since YAML already defines that class.
        update_yaml = False

    if not ask_confirm(
        "Change class IDs of all filled labels?",
        False,
    ):
        print("No changes made.")
        return

    backup = backup_metadata(root, "class_id")
    if backup:
        print("Label/YAML backup:", backup)

    show_progress(
        "Changing label ID",
        0,
        len(plans),
        f"0/{len(plans)} labels",
    )

    for index, (label, text) in enumerate(plans, 1):
        label.write_text(text, encoding="utf-8")

        if (
            index == len(plans)
            or index % max(1, len(plans) // 100) == 0
        ):
            show_progress(
                "Changing label ID",
                index,
                len(plans),
                f"{index}/{len(plans)} labels",
                finish=index == len(plans),
            )

    if update_yaml:
        # Preserve classes in existing data.yaml and place directly at user-provided
        # ID instead of appending to the end.
        updated_names = dict(names)
        updated_names[new_id] = target_name

        write_yaml(
            yaml_path,
            data,
            updated_names,
        )

        print(
            f"data.yaml updated: "
            f"{new_id}: {target_name}"
        )
    else:
        if created_new_class:
            print(
                "data.yaml was not changed. "
                f"New ID {new_id} was written to label files but "
                "not added to data.yaml."
            )
        else:
            print(
                "Existing data.yaml class was used; "
                "no additional changes made to data.yaml."
            )

    # If new ID was not written to data.yaml, validate_dataset() rejects it
    # as unknown class ID. In this case, check only label format and coordinates.
    if update_yaml or not created_new_class:
        validate_dataset(
            root,
            interactive=False,
            require_yaml=True,
        )
    else:
        manifest(
            root,
            require_yaml=False,
            validate_class_ids=False,
        )

    print(
        f"Bulk class ID change completed. "
        f"All filled boxes: {new_id}:{target_name}"
    )


def extract_or_delete_class_pairs(root, data, names):
    selected = set(ask_checkbox(
        "Select the classes you want to filter:",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(names)],
    ))
    if not selected:
        raise ValueError("At least one class must be selected.")

    policy = ask_select(
        "How should images be filtered?",
        [questionary.Choice(
            "An object whose class is not in the new data.yaml can remain in an image "
            "training the model for the selected class (recommended)",
            "allow_other"),
         questionary.Choice(
            "Only images containing boxes of the classes we selected should be used",
            "only_effective")],
    )
    action = ask_select(
        "Operation you want to perform:",
        [questionary.Choice("Copy to another folder", "copy"),
         questionary.Choice("Delete", "delete")],
    )

    destination = None
    if action == "copy":
        destination_parent = choose_directory(
            "Select parent directory where copy will be saved:", BASE_DIRECTORY
        )
        destination_name = ask_text(
            "Name of the new destination dataset folder "
            "(if left blank, train/val/test and data.yaml will be used or created "
            "inside the selected directory):"
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
            raise ValueError("Destination dataset cannot be inside the source dataset.")

    scope = ask_select(
        "Which type should this apply to:",
        [questionary.Choice("Selected classes", "selected"),
         questionary.Choice("Unselected classes", "unselected")],
    )
    effective_ids = selected if scope == "selected" else set(names) - selected
    if not effective_ids:
        raise ValueError("No usable classes remaining as a result of this selection.")

    # Filtered dataset will have sequential IDs starting from 0 within itself.
    filtered_order = [cid for cid in sorted(names) if cid in effective_ids]
    filtered_id_by_old = {old_id: new_id for new_id, old_id in enumerate(filtered_order)}
    destination_data = dict(data)
    destination_names = {}
    destination_yaml = destination / DATA_YAML_NAME if destination is not None else None
    copy_id_map = dict(filtered_id_by_old)

    if action == "copy":
        if (destination / "images").is_dir() or (destination / "labels").is_dir():
            raise RuntimeError(
                "Selected destination appears to be in images/train + labels/train ZIP structure. "
                "This filtering operation uses train/images + train/labels workflow layout."
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
                        f"Source data.yaml to copy has {filtered_id}:{class_name}, destination "
                        f"data.yaml has {target_id}:{output_names[target_id]}. "
                        "To preserve the dataset, should the copied label class ID be set to "
                        f"{target_id}? If No is selected, operation will be cancelled.",
                        True,
                    ):
                        raise SystemExit("Class ID mapping rejected; copy cancelled.")
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
                        f"ID {filtered_id}:{class_name} to copy is used in destination data.yaml by "
                        f"{filtered_id}:{conflicting_name}. Closest free ID is {target_id}. "
                        f"Should filtered {class_name} label IDs be set to "
                        f"{target_id}? If No is selected, operation will be cancelled.",
                        True,
                    ):
                        raise SystemExit("Free class ID suggestion rejected; copy cancelled.")
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
                    "No data.yaml found at destination but existing label files exist. "
                    "Safe copying cannot be performed without knowing Class ID meanings."
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
            raise ValueError(f"No suitable images found for {cid}:{names[cid]}.")
        raw = ask_text(
            f"How many should be filtered for {cid}:{names[cid]}? "
            f"(found={len(candidates)}, 0=all):",
            "0",
        )
        try:
            amount = int(raw)
        except ValueError as exc:
            raise ValueError("Filter amount must be an integer 0 or greater.") from exc
        if amount < 0 or amount > len(candidates):
            raise ValueError(
                f"Amount for {cid}:{names[cid]} must be between 0 and {len(candidates)}."
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
        raise ValueError("No matching image-label pairs found for the selection.")
    mixed_pairs = sum(bool(present - effective_ids) for _, _, present in chosen)

    print("\nClass data operation plan:")
    print("  Source:", root)
    print("  Operation:", "copy to another dataset" if action == "copy" else "delete image-label pairs")
    if destination is not None:
        print("  Destination:", destination)
    print("  Scope:", ", ".join(f"{cid}:{names[cid]}" for cid in sorted(effective_ids)))
    if action == "copy":
        print("  Destination data.yaml class mappings:")
        for old_id in filtered_order:
            filtered_id = filtered_id_by_old[old_id]
            target_id = copy_id_map[old_id]
            print(f"    {filtered_id}:{names[old_id]} -> {target_id}:{destination_names[target_id]}")
    for cid in sorted(effective_ids):
        amount_text = "all" if requested[cid] == 0 else str(requested[cid])
        print(f"  {cid}:{names[cid]} -> requested={amount_text}, available={available[cid]}")
        print(
            "    existing: "
            + ", ".join(f"{split}={split_available[cid][split]}" for split in SPLITS)
        )
        print(
            "    to select: "
            + ", ".join(f"{split}={split_requested[cid][split]}" for split in SPLITS)
        )
    print(f"  Unique pairs to process={len(chosen)}, mixed-class images={mixed_pairs}")
    if len(effective_ids) > 1:
        print(
            "  NOTE: If an image contains multiple target classes, this image can contribute "
            "to each class selection but is copied/deleted only once."
        )
    if action == "copy" and mixed_pairs:
        print(
            "  WARNING: Out-of-scope objects will remain visible in the image but their boxes "
            "will not be written to new labels; they may be treated as background in training."
        )
    if action == "delete" and mixed_pairs:
        print(
            "  WARNING: Some selected candidates also contain out-of-scope classes. Deletion "
            "will completely remove these images and their label files."
        )
    if not ask_confirm("Apply this plan?", False):
        print("No changes made.")
        return

    if action == "delete":
        backup = backup_metadata(root, "class_pair_delete")
        if backup:
            print("Label/YAML backup:", backup)
        total = len(chosen)
        show_progress("Deleting", 0, total, f"0/{total} pairs")
        for index, (pair, _, _) in enumerate(chosen, 1):
            pair.image.unlink()
            pair.label.unlink()
            if index == total or index % max(1, total // 100) == 0:
                show_progress("Deleting", index, total, f"{index}/{total} pairs", finish=index == total)
        validate_dataset(root, False, True)
        print(f"Deletion completed: {total} image-label pairs removed.")
        return

    if destination.exists() and destination_yaml.is_file():
        backup = backup_metadata(destination, "filter_copy")
        if backup:
            print("Destination label/YAML backup:", backup)
    dirs = split_dirs(destination, create=True)
    total = len(chosen)
    show_progress("Copying class data", 0, total, f"0/{total} pairs")
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
                "Copying class data", index, total, f"{index}/{total} pairs",
                finish=index == total,
            )
    write_yaml(destination_yaml, destination_data, destination_names)
    validate_dataset(destination, False, True)
    print(f"Copying completed: pairs={total}, boxes={copied_boxes}, destination={destination}")



def annotate_images_with_boxes():
    """
    Perform YOLO bounding-box annotation using GUI.

    - Shows images from the selected images folder one by one.
    - Reads boxes if existing .txt label exists and displays them in class colors.
    - Draw new rectangles/squares by clicking and dragging with mouse.
    - Select one of the classes found in data.yaml at the top.
    - Use distinct colors for each class.
    - Overwrites current label file on Save; creates it if no label exists.
    """
    if tk is None or Image is None:
        raise RuntimeError(
            "Pillow and Tkinter are required for GUI. "
            "Installation: python -m pip install Pillow"
        )

    selected_path = choose_directory(
        "Select folder containing images to annotate boxes:",
        BASE_DIRECTORY,
    ).resolve()

    selected_name = selected_path.name.casefold()

    if selected_name == "images":
        images_dir = selected_path
        labels_dir = selected_path.parent / "labels"

    elif selected_name in {"train", "val", "valid", "test"}:
        split_name = "val" if selected_name == "valid" else selected_name
        images_candidate = selected_path / "images"
        labels_candidate = selected_path / "labels"

        if images_candidate.is_dir():
            images_dir = images_candidate
            labels_dir = labels_candidate
        else:
            direct_images = [
                p for p in selected_path.iterdir()
                if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS
            ]
            if direct_images:
                images_dir = selected_path
                labels_dir = selected_path.parent / "labels"
            else:
                raise ValueError(
                    "Images folder or images could not be found in selected split folder: "
                    f"{selected_path}"
                )

    elif selected_name == "labels":
        raise ValueError(
            "Labels folder selected. Select images folder for annotation."
        )

    else:
        direct_images = [
            p for p in selected_path.iterdir()
            if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS
        ]

        if direct_images:
            images_dir = selected_path
            if selected_path.parent.name.casefold() == "images":
                labels_dir = selected_path.parent.parent / "labels"
            else:
                labels_dir = selected_path.parent / "labels"
        else:
            # Standard YOLO dataset root:
            # dataset/train/images + dataset/train/labels
            # dataset/val/images + dataset/val/labels
            # dataset/test/images + dataset/test/labels
            candidates = []

            for split in SPLITS:
                candidate = selected_path / split / "images"
                if candidate.is_dir():
                    candidates.append((split, candidate))

            valid_candidate = selected_path / "valid" / "images"
            if valid_candidate.is_dir():
                candidates.append(("val", valid_candidate))

            if not candidates:
                raise ValueError(
                    "Supported images or YOLO split structure could not be found "
                    "in selected folder: "
                    f"{selected_path}"
                )

            if len(candidates) == 1:
                split_name, images_dir = candidates[0]
            else:
                split_name = ask_select(
                    "Which split do you want to annotate?",
                    [
                        questionary.Choice(
                            f"{split}: {candidate}",
                            split,
                        )
                        for split, candidate in candidates
                    ],
                )
                images_dir = dict(candidates)[split_name]

            labels_dir = selected_path / split_name / "labels"
            if not labels_dir.is_dir():
                alternative_labels = selected_path / "labels" / split_name
                if alternative_labels.is_dir():
                    labels_dir = alternative_labels

    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()

    if (
        images_dir.parent.name.casefold() in SPLITS
        and images_dir.parent.parent.is_dir()
    ):
        dataset_root = images_dir.parent.parent
    else:
        dataset_root = images_dir.parent

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images folder not found: {images_dir}"
        )

    # The selected directory may be a dataset root, a split directory, an
    # images directory, or a directory containing images directly. Resolve
    # the final images/labels pair solely from its filesystem structure.
    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()

    images = image_files(images_dir)
    if not images:
        raise FileNotFoundError(
            f"No supported images found in selected directory: {images_dir}"
        )

    # Search for data.yaml in selected directory first, then in parent directory.
    yaml_candidates = [
        images_dir / DATA_YAML_NAME,
        images_dir.parent / DATA_YAML_NAME,
        dataset_root / DATA_YAML_NAME,
        dataset_root.parent / DATA_YAML_NAME,
    ]
    print(
        "Annotation paths resolved:",
        f"\n  Images: {images_dir}",
        f"\n  Labels: {labels_dir}",
        f"\n  Dataset root: {dataset_root}",
        f"\n  YAML candidates: {yaml_candidates}",
    )

    yaml_path = next((p for p in yaml_candidates if p.is_file()), None)
    if yaml_path is None:
        raise FileNotFoundError(
            "data.yaml not found for class list. Searched locations:\n"
            + "\n".join(f"  {p}" for p in yaml_candidates)
        )

    yaml_root = yaml_path.parent
    data, names, _ = load_yaml_for_annotation(yaml_path)
    if not names:
        raise ValueError(f"No classes found in data.yaml: {yaml_path}")

    # If labels folder does not exist, it can be created during annotation.
    labels_dir.mkdir(parents=True, exist_ok=True)

    # HSV colors: ensure colors are distinct regardless of class count.
    import colorsys

    # Automatically choose visually distinct, high-contrast colors.
    # We deliberately avoid adjacent hues so neighboring classes are not
    # easily confused. The palette is generated from a large set of
    # perceptually separated HSV positions and then rotated according to
    # the class order.
    names = {
        int(class_id): str(class_name)
        for class_id, class_name in names.items()
    }
    ordered_ids = sorted(names)

    def build_distinct_class_colors(class_ids):
        palette = [
            (230, 25, 75),    # red
            (60, 180, 75),    # green
            (0, 130, 200),   # blue
            (245, 130, 48),   # orange
            (145, 30, 180),   # purple
            (70, 240, 240),  # cyan
            (240, 50, 230),  # magenta
            (210, 245, 60),  # lime/yellow
            (250, 190, 190), # light red
            (170, 110, 40),  # brown
            (0, 128, 128),   # teal
            (128, 0, 0),     # dark red
            (0, 0, 128),     # dark blue
            (128, 128, 0),   # olive
            (0, 128, 0),     # dark green
            (128, 0, 128),   # dark purple
        ]

        colors = {}
        for index, class_id in enumerate(class_ids):
            if index < len(palette):
                colors[class_id] = palette[index]
                continue

            # For additional classes, generate a hue that is deliberately
            # separated from the already-used colors.
            hue = ((index - len(palette) + 1) * 0.61803398875) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.88, 0.95)
            colors[class_id] = tuple(
                int(channel * 255) for channel in rgb
            )

        return colors

    class_colors = build_distinct_class_colors(ordered_ids)

    def rgb_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def find_label_for_image(image):
        return labels_dir / f"{image.stem}.txt"

    def read_boxes(label_path):
        boxes = []
        if not label_path.is_file():
            return boxes

        try:
            for line_no, raw in enumerate(
                label_path.read_text(encoding="utf-8-sig").splitlines(), 1
            ):
                parts = raw.strip().split()
                if len(parts) != 5:
                    # In this annotation screen, only standard YOLO bbox lines
                    # are displayed; polygons remain unchanged.
                    continue
                try:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                if class_id not in names:
                    continue
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                    continue
                if bw <= 0 or bh <= 0:
                    continue
                boxes.append({
                    "class_id": class_id,
                    "cx": cx,
                    "cy": cy,
                    "w": bw,
                    "h": bh,
                })
        except OSError:
            pass
        return boxes

    root = tk.Tk()
    root.title("YOLO Box Annotation")
    root.geometry("1250x900")
    root.minsize(950, 700)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    state = {
        "index": 0,
        "image": None,
        "photo": None,
        "original_size": None,
        "display_size": None,
        "display_origin": None,
        "boxes": [],
        "dirty": False,
        "drag_start": None,
        "drag_rect": None,
        "selected_box": None,
        "closed": False,
    }

    # ----------------------------- UI -----------------------------
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    ttk.Label(
        top,
        text="Class:",
        font=("TkDefaultFont", 11, "bold"),
    ).pack(side="left")

    class_var = tk.StringVar(value=f"{ordered_ids[0]}: {names[ordered_ids[0]]}")
    class_values = [
        f"{class_id}: {names[class_id]}"
        for class_id in ordered_ids
    ]
    class_combo = ttk.Combobox(
        top,
        textvariable=class_var,
        values=class_values,
        state="readonly",
        width=max(25, min(55, max(len(v) for v in class_values) + 2)),
    )
    class_combo.pack(side="left", padx=(8, 14))

    color_canvas = tk.Canvas(
        top,
        width=28,
        height=28,
        highlightthickness=1,
        highlightbackground="#666666",
    )
    color_canvas.pack(side="left", padx=(0, 14))

    image_info_var = tk.StringVar()
    image_info = ttk.Label(
        top,
        textvariable=image_info_var,
        font=("TkDefaultFont", 10, "bold"),
    )
    image_info.pack(side="left", padx=(8, 0))

    dirty_var = tk.StringVar()
    ttk.Label(
        top,
        textvariable=dirty_var,
        foreground="#cc7700",
    ).pack(side="right")

    viewer_frame = ttk.Frame(root, padding=(10, 0, 10, 8))
    viewer_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(
        viewer_frame,
        background="#202020",
        highlightthickness=1,
        highlightbackground="#666666",
        takefocus=True,
    )
    canvas.pack(fill="both", expand=True)

    bottom = ttk.Frame(root, padding=10)
    bottom.pack(fill="x")

    legend_frame = ttk.Frame(bottom)
    legend_frame.pack(fill="x", pady=(0, 7))

    ttk.Label(
        legend_frame,
        text="Class colors:",
        font=("TkDefaultFont", 9, "bold"),
    ).pack(side="left", padx=(0, 8))

    legend_widgets = []
    for class_id in ordered_ids:
        item = ttk.Frame(legend_frame)
        item.pack(side="left", padx=(0, 12))
        swatch = tk.Canvas(
            item,
            width=15,
            height=15,
            highlightthickness=0,
        )
        swatch.create_rectangle(
            1, 1, 14, 14,
            fill=rgb_hex(class_colors[class_id]),
            outline="",
        )
        swatch.pack(side="left")
        ttk.Label(
            item,
            text=f" {class_id}: {names[class_id]}",
        ).pack(side="left")

    hint_var = tk.StringVar(
        value=(
            "Mouse: draw box | Click existing box: select | "
            "Delete: delete selected box | ←/→: image | Ctrl+S: save | "
            "Enter: save & next | Esc: exit"
        )
    )
    ttk.Label(
        bottom,
        textvariable=hint_var,
        foreground="#666666",
    ).pack(fill="x", pady=(0, 8))

    button_row = ttk.Frame(bottom)
    button_row.pack(fill="x")

    previous_button = ttk.Button(button_row, text="← Previous")
    previous_button.pack(side="left", fill="x", expand=True, padx=(0, 4))

    save_button = ttk.Button(button_row, text="Save")
    save_button.pack(side="left", fill="x", expand=True, padx=4)

    next_button = ttk.Button(button_row, text="Next →")
    next_button.pack(side="left", fill="x", expand=True, padx=4)

    save_next_button = ttk.Button(button_row, text="Save & Next")
    save_next_button.pack(side="left", fill="x", expand=True, padx=4)

    finish_button = ttk.Button(button_row, text="Exit")
    finish_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

    status_var = tk.StringVar()
    ttk.Label(
        bottom,
        textvariable=status_var,
        foreground="#555555",
    ).pack(fill="x", pady=(7, 0))

    # ------------------------- helpers -------------------------
    def selected_class_id():
        value = class_var.get().split(":", 1)[0].strip()
        try:
            class_id = int(value)
        except ValueError:
            class_id = int(ordered_ids[0])

        if class_id not in names:
            raise ValueError(
                f"Selected class ID not found in data.yaml: {class_id}"
            )

        if class_id not in class_colors:
            # This should never be needed after normalization, but keep the
            # GUI fail-safe: every valid class ID must always have a color.
            fallback_index = ordered_ids.index(class_id)
            hue = (fallback_index * 0.61803398875) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.88, 0.95)
            class_colors[class_id] = tuple(
                int(channel * 255) for channel in rgb
            )

        return class_id

    def update_color_indicator():
        class_id = selected_class_id()
        color = rgb_hex(class_colors[class_id])
        color_canvas.delete("all")
        color_canvas.create_rectangle(
            1, 1, 27, 27,
            fill=color,
            outline="",
        )

    def update_dirty():
        dirty_var.set("* Unsaved changes" if state["dirty"] else "")

    def canvas_to_image(x, y):
        if not state["display_size"] or not state["display_origin"]:
            return None
        ox, oy = state["display_origin"]
        dw, dh = state["display_size"]
        if dw <= 0 or dh <= 0:
            return None

        # Convert Canvas coordinates to normalized 0..1 image coordinates.
        nx = (x - ox) / dw
        ny = (y - oy) / dh
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            return None
        return nx, ny

    def image_to_canvas(box):
        if not state["display_size"] or not state["display_origin"]:
            return None
        ox, oy = state["display_origin"]
        dw, dh = state["display_size"]

        x1 = ox + (box["cx"] - box["w"] / 2) * dw
        y1 = oy + (box["cy"] - box["h"] / 2) * dh
        x2 = ox + (box["cx"] + box["w"] / 2) * dw
        y2 = oy + (box["cy"] + box["h"] / 2) * dh
        return x1, y1, x2, y2

    def draw_all():
        canvas.delete("all")
        if state["photo"] is None:
            return

        ox, oy = state["display_origin"]
        dw, dh = state["display_size"]
        canvas.create_image(
            ox + dw / 2,
            oy + dh / 2,
            image=state["photo"],
            anchor="center",
            tags="image",
        )

        # Existing and new boxes are drawn in different colors per class.
        for index, box in enumerate(state["boxes"]):
            coords = image_to_canvas(box)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            class_id = box["class_id"]
            color = rgb_hex(class_colors[class_id])
            selected = state["selected_box"] == index

            canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="#ffffff" if selected else color,
                width=5 if selected else 3,
                tags=("box", f"box_{index}"),
            )

            # Draw class text label at top-left of box.
            label_text = f"{class_id}: {names[class_id]}"
            text_id = canvas.create_text(
                x1 + 5,
                max(5, y1 + 5),
                text=label_text,
                anchor="nw",
                fill="#ffffff",
                font=("TkDefaultFont", 10, "bold"),
                tags=("box", f"box_{index}"),
            )

            # Background rectangle for text readability.
            bbox = canvas.bbox(text_id)
            if bbox:
                background = canvas.create_rectangle(
                    bbox[0] - 2, bbox[1] - 1,
                    bbox[2] + 2, bbox[3] + 1,
                    fill=color,
                    outline="",
                    tags=("box", f"box_{index}"),
                )
                canvas.tag_lower(background, text_id)

        update_dirty()

    def load_current():
        image = images[state["index"]]
        label = find_label_for_image(image)

        try:
            with Image.open(image) as im:
                original = im.convert("RGB")
                original_size = original.size

            # update_idletasks first to ensure canvas dimensions are ready.
            root.update_idletasks()
            max_w = max(300, canvas.winfo_width() - 30)
            max_h = max(250, canvas.winfo_height() - 30)

            display = original.copy()
            display.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            state["image"] = image
            state["original_size"] = original_size
            state["display_size"] = display.size
            state["photo"] = ImageTk.PhotoImage(display)

            ox = (canvas.winfo_width() - display.size[0]) / 2
            oy = (canvas.winfo_height() - display.size[1]) / 2
            state["display_origin"] = (ox, oy)

            state["boxes"] = read_boxes(label)
            state["selected_box"] = None
            state["drag_start"] = None
            state["drag_rect"] = None
            state["dirty"] = False

            image_info_var.set(
                f"{state['index'] + 1}/{len(images)}  |  "
                f"{image.name}  |  {original_size[0]}x{original_size[1]}  |  "
                f"Boxes: {len(state['boxes'])}"
            )
            status_var.set(
                f"Label: {label if label.is_file() else 'none (will be created on Save)'}"
            )
            update_color_indicator()
            draw_all()
        except Exception as exc:
            state["image"] = image
            state["photo"] = None
            state["boxes"] = []
            state["dirty"] = False
            canvas.delete("all")
            canvas.create_text(
                20, 20,
                anchor="nw",
                fill="white",
                text=f"Failed to open image:\n{image}\n\n{exc}",
            )
            status_var.set(f"ERROR: {exc}")

    def save_current():
        image = images[state["index"]]
        label = find_label_for_image(image)

        # Write standard YOLO bboxes only.
        lines = []
        for box in state["boxes"]:
            lines.append(
                f"{box['class_id']} "
                f"{box['cx']:.6f} "
                f"{box['cy']:.6f} "
                f"{box['w']:.6f} "
                f"{box['h']:.6f}"
            )

        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        state["dirty"] = False
        status_var.set(f"Saved: {label}")
        image_info_var.set(
            f"{state['index'] + 1}/{len(images)}  |  "
            f"{image.name}  |  {state['original_size'][0]}x{state['original_size'][1]}  |  "
            f"Boxes: {len(state['boxes'])}"
        )
        update_dirty()

    def ensure_saved_before_navigation():
        if not state["dirty"]:
            return True

        answer = questionary.confirm(
            "There are unsaved boxes in the current image. Proceed without saving?",
            default=False,
        ).ask()
        if answer is None:
            return False
        if answer:
            return True

        save_current()
        return True

    def move(delta, save_if_dirty=False):
        if state["dirty"]:
            if save_if_dirty:
                save_current()
            elif not ensure_saved_before_navigation():
                return

        new_index = state["index"] + delta
        if 0 <= new_index < len(images):
            state["index"] = new_index
            load_current()

    # ----------------------- mouse drawing -----------------------
    def on_mouse_down(event):
        canvas.focus_set()

        # If an existing box is clicked, select it.
        hits = canvas.find_overlapping(event.x, event.y, event.x, event.y)
        hit_indices = []
        for item_id in hits:
            tags = canvas.gettags(item_id)
            for tag in tags:
                if tag.startswith("box_"):
                    try:
                        hit_indices.append(int(tag.split("_", 1)[1]))
                    except ValueError:
                        pass

        if hit_indices:
            state["selected_box"] = hit_indices[-1]
            draw_all()
            status_var.set(
                "Existing box selected. Press Delete to delete, or select class at the top "
                "and press R to reassign class."
            )
            return

        point = canvas_to_image(event.x, event.y)
        if point is None:
            return

        state["selected_box"] = None
        state["drag_start"] = point
        if state["drag_rect"] is not None:
            canvas.delete(state["drag_rect"])
            state["drag_rect"] = None

    def on_mouse_move(event):
        start = state["drag_start"]
        current = canvas_to_image(event.x, event.y)
        if start is None or current is None:
            return

        # Draw strictly within image bounds.
        x1n, y1n = start
        x2n, y2n = current
        x1 = min(x1n, x2n)
        y1 = min(y1n, y2n)
        x2 = max(x1n, x2n)
        y2 = max(y1n, y2n)

        ox, oy = state["display_origin"]
        dw, dh = state["display_size"]
        cx1 = ox + x1 * dw
        cy1 = oy + y1 * dh
        cx2 = ox + x2 * dw
        cy2 = oy + y2 * dh

        if state["drag_rect"] is not None:
            canvas.delete(state["drag_rect"])

        class_id = selected_class_id()
        color = rgb_hex(class_colors[class_id])
        state["drag_rect"] = canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline=color,
            width=3,
            dash=(7, 4),
        )

    def on_mouse_up(event):
        start = state["drag_start"]
        current = canvas_to_image(event.x, event.y)
        state["drag_start"] = None

        if start is None or current is None:
            if state["drag_rect"] is not None:
                canvas.delete(state["drag_rect"])
                state["drag_rect"] = None
            return

        x1n, y1n = start
        x2n, y2n = current
        x1 = max(0.0, min(1.0, min(x1n, x2n)))
        y1 = max(0.0, min(1.0, min(y1n, y2n)))
        x2 = max(0.0, min(1.0, max(x1n, x2n)))
        y2 = max(0.0, min(1.0, max(y1n, y2n)))

        if state["drag_rect"] is not None:
            canvas.delete(state["drag_rect"])
            state["drag_rect"] = None

        # Prevent accidental creation of click-sized tiny boxes.
        if (x2 - x1) < 0.005 or (y2 - y1) < 0.005:
            return

        class_id = selected_class_id()
        state["boxes"].append({
            "class_id": class_id,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "w": x2 - x1,
            "h": y2 - y1,
        })
        state["selected_box"] = len(state["boxes"]) - 1
        state["dirty"] = True
        status_var.set(
            f"New box added: {class_id}: {names[class_id]}. "
            "Use Ctrl+S or Save button to save."
        )
        draw_all()

    def delete_selected_box():
        index = state["selected_box"]
        if index is None or not (0 <= index < len(state["boxes"])):
            status_var.set("No selected box to delete.")
            return
        deleted = state["boxes"].pop(index)
        state["selected_box"] = None
        state["dirty"] = True
        status_var.set(
            f"Box deleted: {deleted['class_id']}: {names[deleted['class_id']]}. "
            "Changes not saved yet."
        )
        draw_all()

    def reclass_selected_box():
        index = state["selected_box"]
        if index is None or not (0 <= index < len(state["boxes"])):
            status_var.set("No selected box to change class.")
            return
        old_id = state["boxes"][index]["class_id"]
        new_id = selected_class_id()
        state["boxes"][index]["class_id"] = new_id
        state["dirty"] = True
        status_var.set(
            f"Box class changed: {old_id}:{names[old_id]} -> "
            f"{new_id}:{names[new_id]}. Not saved."
        )
        draw_all()

    def finish():
        if state["dirty"]:
            save_current()
        state["closed"] = True
        root.quit()

    def cancel():
        state["closed"] = True
        root.quit()

    def on_class_changed(_event=None):
        update_color_indicator()
        status_var.set(
            f"New drawing box class: {selected_class_id()}: "
            f"{names[selected_class_id()]}"
        )

    def on_key(event):
        if event.keysym == "Left":
            move(-1)
            return "break"
        if event.keysym == "Right":
            move(1)
            return "break"
        if event.keysym == "Delete":
            delete_selected_box()
            return "break"
        if event.keysym.lower() == "r":
            reclass_selected_box()
            return "break"
        if event.keysym.lower() == "s" and (event.state & 0x4):
            save_current()
            return "break"
        if event.keysym in ("Return", "KP_Enter"):
            # Enter: save and proceed to next image.
            if state["index"] < len(images) - 1:
                save_current()
                state["index"] += 1
                load_current()
            else:
                finish()
            return "break"
        if event.keysym == "Escape":
            cancel()
            return "break"
        return None

    class_combo.bind("<<ComboboxSelected>>", on_class_changed)

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    previous_button.configure(command=lambda: move(-1))
    save_button.configure(command=save_current)
    next_button.configure(command=lambda: move(1))
    save_next_button.configure(
        command=lambda: (
            save_current(),
            move(1, save_if_dirty=False)
        )
    )
    finish_button.configure(command=finish)

    root.bind_all("<KeyPress-Left>", on_key, add="+")
    root.bind_all("<KeyPress-Right>", on_key, add="+")
    root.bind_all("<KeyPress-Delete>", on_key, add="+")
    root.bind_all("<KeyPress-Return>", on_key, add="+")
    root.bind_all("<KeyPress-KP_Enter>", on_key, add="+")
    root.bind_all("<KeyPress-Escape>", on_key, add="+")
    root.bind_all("<Control-KeyPress-s>", on_key, add="+")
    root.bind_all("<KeyPress-r>", on_key, add="+")

    def on_resize(_event=None):
        # Resize image and boxes to fit canvas when window dimensions change.
        if state["image"] is not None:
            current_index = state["index"]
            dirty = state["dirty"]
            boxes = list(state["boxes"])

            try:
                with Image.open(state["image"]) as im:
                    original = im.convert("RGB")
                root.update_idletasks()
                max_w = max(300, canvas.winfo_width() - 30)
                max_h = max(250, canvas.winfo_height() - 30)
                display = original.copy()
                display.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                state["display_size"] = display.size
                state["photo"] = ImageTk.PhotoImage(display)
                state["display_origin"] = (
                    (canvas.winfo_width() - display.size[0]) / 2,
                    (canvas.winfo_height() - display.size[1]) / 2,
                )
                state["boxes"] = boxes
                state["dirty"] = dirty
                draw_all()
            except Exception:
                pass

    # Debounce resize with Tkinter's after mechanism so we don't reload on every single pixel.
    resize_job = {"id": None}

    def schedule_resize(event=None):
        if resize_job["id"] is not None:
            try:
                root.after_cancel(resize_job["id"])
            except Exception:
                pass
        resize_job["id"] = root.after(120, on_resize)

    canvas.bind("<Configure>", schedule_resize)

    root.protocol("WM_DELETE_WINDOW", cancel)

    update_color_indicator()
    load_current()
    root.mainloop()

    if not state["closed"]:
        return

    # If window is closed, ensure unsaved modifications are preserved.
    if state["dirty"]:
        save_current()

    print(
        f"Annotation window closed. Processed image folder: {images_dir}\n"
        f"Label folder: {labels_dir}\n"
        f"Class source: {yaml_path}"
    )

def _impl_filter_classes():
    filter_action = ask_select(
        "Class operation:",
        [questionary.Choice("(1) Keep or remove classes", "filter"),
         questionary.Choice("(2) Merge multiple classes into a single class", "merge_classes"),
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
        raise ValueError("New folder name cannot be empty and cannot contain a path.")
    if any(char in name for char in '<>:"/\\|?*') or name.endswith((" ", ".")):
        raise ValueError("New folder name contains invalid characters for Windows/Linux.")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                *(f"lpt{i}" for i in range(1, 10))}
    if name.casefold() in reserved:
        raise ValueError("This folder name is reserved by Windows.")
    return name



def is_flat_images_labels_dataset(root):
    """Returns True if source has no train/val/test, only images + labels."""
    if not root.is_dir():
        return False
    has_images = (root / "images").is_dir()
    has_labels = (root / "labels").is_dir()
    has_split = any(
        (root / split).is_dir()
        for split in ("train", "val", "valid", "test")
    )
    return has_images and has_labels and not has_split


def choose_flat_destination_layout():
    """
    Select destination layout when merging a flat images+labels source:
      1) train/val/test/images + labels
      2) images + labels
      3) copy directly into selected destination directory
    """
    return ask_select(
        "Destination has no train/val/test folders. How should flat source be placed?",
        [
            questionary.Choice(
                "Create train/val/test folders; place files in train",
                "split",
            ),
            questionary.Choice(
                "Create images and labels folders inside destination",
                "flat",
            ),
            questionary.Choice(
                "Paste directly into selected destination directory",
                "direct",
            ),
        ],
    )



def choose_merge_sources(message):
    """
    Allows selecting both dataset root and a single split folder
    (train/val/valid/test) as source for merge.

    IMPORTANT:
      When a folder like dataset/train or dataset/valid is selected,
      having images + labels inside it designates it as a SPLIT.
      looks_like_dataset() check is NOT performed first; otherwise train/val/valid/test
      folder could mistakenly be assumed as dataset root and new train/val/test
      folders could be created inside it by split_dirs(..., create=True).
    """
    selected = []
    while True:
        selected_path = choose_directory(message, BASE_DIRECTORY).resolve()

        source_root = None
        selected_split = None
        name = selected_path.name.casefold()

        # 1) FIRST single split check.
        # dataset/train, dataset/val, dataset/valid, dataset/test
        if (
            name in {"train", "val", "valid", "test"}
            and (selected_path / "images").is_dir()
            and (selected_path / "labels").is_dir()
        ):
            source_root = selected_path.parent
            selected_split = "val" if name == "valid" else name

        # 2) dataset/images/train, dataset/images/val, ...
        elif (
            selected_path.parent.name.casefold() == "images"
            and name in {"train", "val", "valid", "test"}
            and (
                selected_path.parent.parent / "labels" / name
            ).is_dir()
        ):
            source_root = selected_path.parent.parent
            selected_split = "val" if name == "valid" else name

        # 3) dataset/images + dataset/labels
        elif (
            name == "images"
            and (selected_path.parent / "labels").is_dir()
        ):
            source_root = selected_path.parent

        # 4) Only after these check real dataset root.
        elif looks_like_dataset(selected_path):
            source_root = selected_path

        else:
            raise ValueError(
                "Selected folder must be dataset root, dataset/train|val|valid|test, "
                "images, or images/train|val|valid|test structure."
            )

        entry = (source_root, selected_split)
        if entry in selected:
            raise ValueError("This source has already been selected.")

        selected.append(entry)

        if selected_split is None:
            split_text = "all splits"
        else:
            split_text = f"only {selected_split}"

        print(f"Selected ({len(selected)}): {source_root} -> {split_text}")

        if not ask_confirm("Add another source folder to this group?", False):
            return selected



def manifest_selected_split_without_yaml(root, split):
    """Scan image-label pairs of a single split without data.yaml."""
    dirs = split_dirs(root, create=False)
    paths = dirs[split]
    if not paths["images"].is_dir() or not paths["labels"].is_dir():
        raise FileNotFoundError(
            f"{split}/images and {split}/labels not found in {root}."
        )

    images = image_files(paths["images"])
    show_progress(
        f"Scanning {root.name}/{split}",
        0,
        len(images),
        f"0/{len(images)} images",
    )

    result = []
    for index, image in enumerate(images, 1):
        label = paths["labels"] / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Label not found: {label}")

        # Since there is no YAML, validate class IDs numerically only.
        lines, counts = parse_label(label, None)
        result.append(
            Pair(
                image,
                label,
                split,
                frozenset(counts),
                counts,
            )
        )

        if index == len(images) or index % max(1, len(images) // 100) == 0:
            show_progress(
                f"Scanning {root.name}/{split}",
                index,
                len(images),
                f"{index}/{len(images)} images",
                finish=index == len(images),
            )
    return result


def manifest_selected_split(root, split):
    """Converts only the selected train/val/test split into a Pair list."""
    if split is None:
        return manifest(root, require_yaml=True)

    dirs = split_dirs(root, create=False)
    paths = dirs[split]
    if not paths["images"].is_dir() or not paths["labels"].is_dir():
        raise FileNotFoundError(
            f"{split}/images and {split}/labels not found in {root}."
        )

    _, names, _ = load_yaml(root, required=True)
    known = set(names)
    images = image_files(paths["images"])

    show_progress(
        f"Scanning {root.name}/{split}",
        0,
        len(images),
        f"0/{len(images)} images",
    )
    result = []
    for index, image in enumerate(images, 1):
        label = paths["labels"] / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Label not found: {label}")
        _, counts = parse_label(label, known)
        result.append(
            Pair(
                image,
                label,
                split,
                frozenset(counts),
                counts,
            )
        )
        if index == len(images) or index % max(1, len(images) // 100) == 0:
            show_progress(
                f"Scanning {root.name}/{split}",
                index,
                len(images),
                f"{index}/{len(images)} images",
                finish=index == len(images),
            )
    return result



def resolve_images_labels_target(selected_path):
    """
    Resolve exactly the user's selected target.

    Cases:
      A) selected_path == .../images and sibling .../labels exists:
         images -> selected_path, labels -> sibling labels.
      B) selected_path contains images/ and labels/:
         images -> selected_path/images, labels -> selected_path/labels.
      C) selected_path is train/val/valid/test and contains images/labels:
         images -> selected_path/images, labels -> selected_path/labels.
      D) selected_path is empty/other:
         return None; caller may use the normal dataset workflow.

    IMPORTANT: this function NEVER creates train/val/test.
    """
    selected_path = selected_path.resolve()

    # User selected the actual images directory.
    if selected_path.name.casefold() == "images":
        sibling_labels = selected_path.parent / "labels"
        if sibling_labels.is_dir():
            return {
                "root": selected_path.parent,
                "images": selected_path,
                "labels": sibling_labels,
                "mode": "direct",
            }

        # labels does not exist: caller must ask before creating it.
        return {
            "root": selected_path.parent,
            "images": selected_path,
            "labels": sibling_labels,
            "mode": "direct_missing_labels",
        }

    # User selected a directory that already contains images + labels.
    if (
        (selected_path / "images").is_dir()
        and (selected_path / "labels").is_dir()
    ):
        return {
            "root": selected_path,
            "images": selected_path / "images",
            "labels": selected_path / "labels",
            "mode": "direct",
        }

    return None



def _impl_merge_datasets():
    sources = choose_merge_sources(
        "Select SOURCE folder/split to copy into main dataset:"
    )
    if not sources:
        raise ValueError("At least one source dataset must be selected.")

    source_roots = {root for root, _ in sources}
    source_info = []
    for source, selected_split in sources:
        # If only train/val/valid/test is selected, allow continuing without data.yaml.
        # Since class ID mapping cannot be performed, source class names are accepted
        # as empty in this case and target selection proceeds.
        if selected_split is not None:
            yaml_path = source / DATA_YAML_NAME
            if not yaml_path.is_file():
                if not ask_confirm(
                    f"data.yaml not found in {source}. "
                    "Do you want to continue without data.yaml?",
                    False,
                ):
                    raise RuntimeError(
                        f"Operation cancelled because data.yaml was not found: {source}"
                    )

                source_names = {}
                pairs = manifest_selected_split_without_yaml(source, selected_split)
                source_info.append((source, selected_split, source_names, pairs))
                continue

        # If entire dataset root is selected, data.yaml is mandatory.
        if selected_split is None:
            repair_missing_pairs(source)
        # If selected_split != None, new train/val/test folders will NEVER
        # be created in source folder.

        try:
            _, source_names, _ = load_yaml(source)
        except ValueError as exc:
            # If data.yaml exists but names/class list is empty or invalid,
            # give user chance to continue only when single split is selected.
            if selected_split is None:
                raise

            if not ask_confirm(
                f"Class names in {source / DATA_YAML_NAME} are empty or invalid. "
                "Do you want to continue without data.yaml class names?",
                False,
            ):
                raise RuntimeError(
                    f"Operation cancelled because data.yaml class names are invalid: {source}"
                )

            source_names = {}
            pairs = manifest_selected_split_without_yaml(source, selected_split)
            source_info.append((source, selected_split, source_names, pairs))
            continue

        pairs = manifest_selected_split(source, selected_split)
        source_info.append((source, selected_split, source_names, pairs))

    create_new = ask_confirm("Create a new destination/main dataset folder?", False)
    destination_is_new = False
    rebuild_destination_yaml = False
    if create_new:
        destination_parent = choose_directory(
            "Select parent directory where new destination folder will be created:", BASE_DIRECTORY
        )
        destination_name = validate_new_dataset_name(ask_text("Name of new destination folder:"))
        destination = destination_parent / destination_name
        if destination.exists():
            raise FileExistsError(f"A file/folder with this name already exists: {destination}")
        destination_data, destination_names = {}, {}
        destination_yaml = destination / DATA_YAML_NAME
        destination_is_new = True
        rebuild_destination_yaml = True
    else:
        destination = choose_destination_folder(
            "Existing destination/main dataset:",
            source_roots,
        )

        # FIRST: resolve user selection with images+labels logic.
        # train/val/test is NOT CREATED in this block.
        direct_target = resolve_images_labels_target(destination)

        if direct_target is not None:
            flat_destination_mode = "direct"
            destination_images = direct_target["images"]
            destination_labels = direct_target["labels"]

            if direct_target["mode"] == "direct_missing_labels":
                if not ask_confirm(
                    f"No labels folder in the same directory as {destination_images}. "
                    "Do you want to create labels folder and copy label files here?",
                    False,
                ):
                    raise RuntimeError(
                        "Operation halted because labels folder does not exist."
                    )
                destination_labels.mkdir(parents=True, exist_ok=True)

            print(
                f"Target images : {destination_images}\n"
                f"Target labels : {destination_labels}\n"
                "train/val/test folders will NOT BE CREATED inside this destination."
            )

            # In flat/direct destinations, split_dirs and repair_missing_pairs
            # are strictly not used. Only selected images/labels pair is used.
            destination_yaml = direct_target["root"] / DATA_YAML_NAME
            yaml_exists = destination_yaml.is_file()

            destination_data, destination_names, _ = load_yaml(
                direct_target["root"], required=False
            )

            if yaml_exists:
                rebuild_destination_yaml = ask_confirm(
                    "data.yaml already exists at destination. Rebuild according to "
                    "data.yaml class order of selected sources?",
                    False,
                )
            else:
                # If existing image/label pairs exist at destination, do not use manifest();
                # that function may assume split structure. Scan folders directly.
                existing_images = image_files(destination_images)
                existing_labels = list(destination_labels.glob("*.txt"))
                if existing_images or existing_labels:
                    if not destination_names:
                        print(
                            "No data.yaml at destination and existing images/labels files exist. "
                            "Existing ID meanings will be preserved; new data.yaml will only be written "
                            "if class names can be determined."
                        )
                rebuild_destination_yaml = True

        else:
            # Reached only when user selects a real dataset root / normal target.
            # Flat/split target is not modified.
            flat_destination_mode = None

            destination_data, destination_names, destination_yaml = load_yaml(
                destination, required=False
            )

            # Missing split structure can be created here in a normal target.
            # This NEVER runs for single images/labels selection.
            required_paths = [
                destination / split / kind
                for split in SPLITS for kind in ("images", "labels")
            ]
            if not all(path.is_dir() for path in required_paths):
                split_dirs(destination, create=True)

            repair_missing_pairs(destination)

            yaml_exists = destination_yaml.is_file()
            if yaml_exists:
                rebuild_destination_yaml = ask_confirm(
                    "data.yaml already exists at destination. Rebuild according to "
                    "data.yaml class order of selected sources? Existing label IDs "
                    "will also be safely updated.",
                    False,
                )
            else:
                existing_pairs = manifest(destination, require_yaml=False)
                if existing_pairs:
                    raise RuntimeError(
                        "No data.yaml at destination but existing images/labels found. "
                        "Automatic data.yaml creation is unsafe without knowing old ID meanings."
                    )
                rebuild_destination_yaml = True
                print(
                    "No data.yaml at destination; will be created from "
                    "selected source data.yaml files."
                )


    mode_choices = [questionary.Choice(
        "Match by data.yaml names; append new ones to end (recommended)", "names")]
    if not destination_is_new and not rebuild_destination_yaml:
        mode_choices.extend([
            questionary.Choice(
                "Map all filled boxes in each source to a single selected target ID", "single"),
            questionary.Choice(
                "Copy IDs without changing (only if exactly identical meanings)", "raw"),
        ])
    mode = ask_select(
        "Class ID mapping method:",
        mode_choices,
    )

    output_names = {} if rebuild_destination_yaml else dict(destination_names)
    mappings = {}
    for source, selected_split, source_names, source_pairs in source_info:
        source_key = (source, selected_split)
        if mode == "names":
            if not source_names:
                # Class names are unknown in single-split source without data.yaml.
                # In this case, copying label IDs as-is is the only safe behavior;
                # used if relevant IDs already exist in target YAML.
                if output_names:
                    max_source_id = max(
                        (cid for pair in source_pairs for cid in pair.class_ids),
                        default=-1,
                    )
                    missing_ids = [
                        cid for cid in range(max_source_id + 1)
                        if cid not in output_names
                    ]
                    if missing_ids:
                        raise ValueError(
                            f"{source.name}/{selected_split}: no data.yaml and "
                            f"target data.yaml does not define these class IDs: "
                            f"{missing_ids}. Safe mapping cannot be done without knowing class names."
                        )
                id_map = {}
                for pair in source_pairs:
                    for cid in pair.class_ids:
                        id_map[cid] = cid
            else:
                id_map, output_names = build_name_map(source_names, output_names)
        elif mode == "single":
            print(f"\nTarget classes (for {source.name}/{selected_split or 'all splits'}):")
            for cid in sorted(output_names):
                print(f"  {cid}: {output_names[cid]}")
            new_id = int(ask_text(
                f"Target class ID for all boxes in "
                f"{source.name}/{selected_split or 'all splits'}:"
            ))
            if new_id not in output_names:
                raise ValueError(f"ID does not exist in target data.yaml: {new_id}")
            id_map = {old: new_id for old in source_names}
        else:
            if set(source_names) - set(output_names):
                raise ValueError(f"{source.name}: contains ID not present in target; RAW unsafe.")
            bad = [
                i for i in source_names
                if normalized_name(source_names[i]) != normalized_name(output_names[i])
            ]
            if bad:
                raise ValueError(f"{source.name}: same ID has different class meaning: {bad}")
            id_map = {i: i for i in source_names}
        mappings[source_key] = id_map

    destination_id_map = None
    destination_pairs_before_merge = []
    if rebuild_destination_yaml and not destination_is_new and destination_names:
        destination_pairs_before_merge = manifest(destination, require_yaml=True)
        target_by_name = {
            normalized_name(name): cid for cid, name in output_names.items()
        }
        used_old_ids = set()
        for pair in destination_pairs_before_merge:
            used_old_ids.update(pair.class_ids)
        destination_id_map = {}
        missing_used_classes = []
        for old_id in sorted(used_old_ids):
            key = normalized_name(destination_names[old_id])
            if key not in target_by_name:
                missing_used_classes.append(
                    f"{old_id}:{destination_names[old_id]}"
                )
            else:
                destination_id_map[old_id] = target_by_name[key]
        if missing_used_classes:
            raise RuntimeError(
                "Destination data.yaml cannot be rebuilt. Following classes used "
                "in existing labels are missing in selected source data.yaml files: "
                + ", ".join(missing_used_classes)
            )

    print("\nClass mappings:")
    total_pairs = 0
    for source, selected_split, source_names, pairs in source_info:
        split_text = selected_split or "train/val/test"
        print(f"  [{source.name} -> {split_text}] image-label pairs={len(pairs)}")
        total_pairs += len(pairs)
        for old in sorted(mappings[(source, selected_split)]):
            new = mappings[(source, selected_split)][old]
            print(f"    {old}:{source_names[old]} -> {new}:{output_names[new]}")

    print("\nResulting main data.yaml class order:")
    for cid in sorted(output_names):
        print(f"  {cid}: {output_names[cid]}")
    print(f"\nDestination: {destination}\nTotal pairs to copy: {total_pairs}")
    if not ask_confirm("Start merging?", False):
        return

    if destination_is_new:
        split_dirs(destination, create=True)
    else:
        backup = backup_metadata(destination, "merge")
        if backup:
            print("Destination label/YAML backup:", backup)

    remapped_existing = 0
    if destination_id_map is not None:
        for pair in destination_pairs_before_merge:
            lines, _ = parse_label(pair.label, set(destination_names))
            new_text = rewrite_label(lines, destination_id_map)
            old_text = pair.label.read_text(encoding="utf-8-sig")
            if new_text != old_text:
                pair.label.write_text(new_text, encoding="utf-8")
                remapped_existing += 1
        print(
            f"Existing destination labels updated for new data.yaml order: "
            f"{remapped_existing}"
        )

    # In direct images/labels destination, split directories are not created.
    if not destination_is_new and flat_destination_mode == "direct":
        dirs = None
    elif destination_is_new:
        dirs = split_dirs(destination, create=True)
    else:
        dirs = split_dirs(destination, create=True)

    renamed = negatives = 0
    copied = 0
    copy_progress_step = max(1, total_pairs // 100)
    show_progress("Copying", 0, total_pairs, f"0/{total_pairs} pairs")

    for source, selected_split, source_names, pairs in source_info:
        id_map = mappings[(source, selected_split)]
        for pair in pairs:
            if (
                not destination_is_new
                and flat_destination_mode == "direct"
            ):
                target = {
                    "images": destination_images,
                    "labels": destination_labels,
                }
            else:
                target = dirs[pair.source_split]


            stem, changed = unique_stem(
                pair.image.stem,
                source.name,
                target["images"],
                target["labels"],
            )
            lines, _ = parse_label(pair.label, set(source_names))
            shutil.copy2(
                pair.image,
                target["images"] / f"{stem}{pair.image.suffix}",
            )
            text_label = rewrite_label(lines, id_map)
            destination_label = target["labels"] / f"{stem}.txt"
            original_text = pair.label.read_text(encoding="utf-8-sig")
            if text_label == original_text:
                shutil.copy2(pair.label, destination_label)
            else:
                destination_label.write_text(text_label, encoding="utf-8")

            copied += 1
            renamed += changed
            negatives += not text_label.strip()
            if copied % copy_progress_step == 0 or copied == total_pairs:
                show_progress(
                    "Copying",
                    copied,
                    total_pairs,
                    f"{copied}/{total_pairs} pairs",
                    finish=copied == total_pairs,
                )

    print("Writing data.yaml and validating merged dataset...")
    if output_names:
        write_yaml(destination_yaml, destination_data, output_names)
    else:
        # If class names are unknown, continue without YAML instead of creating
        # empty/erroneous data.yaml. User already approved this situation earlier.
        print("Class names unknown; empty data.yaml will not be created.")

    if not destination_is_new and flat_destination_mode == "direct":
        # Validate directly selected images/labels target.
        validate_flat_dataset(
            direct_target["root"],
            False,
            bool(output_names),
        )
    else:
        validate_dataset(destination, False, bool(output_names))
    print(
        f"Done: copied={copied}, renamed={renamed}, "
        f"negative={negatives}"
    )


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
    ratios = tuple(float(ask_text(f"{name.capitalize()} percentage:", str(int(default))))
                   for name, default in zip(SPLITS, DEFAULT_RATIOS))
    if any(x < 0 or x > 100 for x in ratios) or not math.isclose(sum(ratios), 100, abs_tol=1e-7):
        raise ValueError("Ratios must be between 0-100 and sum up to exactly 100%.")
    return ratios


def _impl_redistribute_datasets():
    declared = int(ask_text("How many class/dataset folders do you have?", "1"))
    if declared < 1:
        raise ValueError("Number of folders must be at least 1.")
    single = choose_many_datasets(
        "Select dataset folders containing single class:", allow_empty=True
    )
    multi = choose_many_datasets(
        "Select dataset folders containing multiple classes:", exclude=set(single), allow_empty=True
    )
    selected = single + multi
    if len(selected) != declared:
        raise ValueError(f"Declared {declared}, but selected {len(selected)} folders.")
    ratios = ask_ratios()
    plans = []
    for root in selected:
        repair_missing_pairs(root)
        pairs = manifest(root, root in multi)
        split_dirs(root, create=True)
        assigned, targets, class_targets = multilabel_assignment(pairs, ratios, root in multi, SEED)
        print(f"\n{root.name}: total={len(pairs)}, target={targets}")
        if root in multi:
            _, names, _ = load_yaml(root)
            for cid in sorted(class_targets):
                print(f"  {cid}:{names[cid]} image target -> {class_targets[cid]}")
        moves = sum(p.source_split != s for s, items in assigned.items() for p in items)
        print("  Pairs to move:", moves)
        plans.append((root, assigned, root in multi))
    if not ask_confirm("Apply redistribution plans?", False):
        return
    for root, assigned, is_multi in plans:
        backup = backup_metadata(root, "split")
        if backup:
            print(f"{root.name} label/YAML backup: {backup}")
        temp = root / f"_split_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        temp.mkdir()
        staged = []
        total = sum(len(items) for items in assigned.values())
        show_progress("Redistribution - temporary move", 0, total, f"0/{total} pairs")
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
                        f"{done}/{total} pairs", finish=done == total,
                    )
        dirs = split_dirs(root, create=True)
        show_progress("Redistribution - placement", 0, total, f"0/{total} pairs")
        for index, (split, old_stem, suffix, ti, tl) in enumerate(staged, 1):
            target = dirs[split]
            stem, _ = unique_stem(old_stem, root.name, target["images"], target["labels"])
            shutil.move(str(ti), target["images"] / f"{stem}{suffix}")
            shutil.move(str(tl), target["labels"] / f"{stem}.txt")
            if index == total or index % max(1, total // 100) == 0:
                show_progress(
                    "Redistribution - placement", index, total,
                    f"{index}/{total} pairs", finish=index == total,
                )
        temp.rmdir()
        validate_dataset(root, False, is_multi)
    print("Proportional redistribution completed.")


def redistribute_datasets():
    try:
        return _impl_redistribute_datasets()
    except BackMenu:
        return None


def conversion_candidates():
    """Lists folders in old split/images layout for conversion."""
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
    """Checks all image-label pairs and labels before conversion."""
    if (root / "val").exists() and (root / "valid").exists():
        raise RuntimeError("Both val and valid folders exist. Select one and consolidate into a single name.")
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
            raise RuntimeError(f"Multiple images with the same base name found: {images_dir}")
        label_stems = {p.stem for p in labels}
        missing_labels = set(image_stems) - label_stems
        missing_images = label_stems - set(image_stems)
        if missing_labels or missing_images:
            raise RuntimeError(
                f"{source_split} mapping error; images missing labels={sorted(missing_labels)[:10]}, "
                f"labels missing images={sorted(missing_images)[:10]}"
            )
        boxes = negatives = 0
        for label in labels:
            lines, _ = parse_label(label, known)
            boxes += len(lines)
            negatives += int(not lines)
        summary[final_split] = (len(images), len(labels), boxes, negatives)
    return aliases, summary


def validate_final_layout(root):
    """Check ZIP-ready structure in images/split + labels/split format."""
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
                f"{split}: image-label mismatch; without label={sorted(image_stems-label_stems)[:10]}, "
                f"without image={sorted(label_stems-image_stems)[:10]}"
            )
        boxes = negatives = 0
        for label in labels:
            lines, _ = parse_label(label, known)
            boxes += len(lines)
            negatives += int(not lines)
        print(
            f"  {split:5s}: images={len(images)}, labels={len(labels)}, "
            f"boxes={boxes}, negatives={negatives} [OK]"
        )


def _impl_convert_to_zip_layout():
    root = choose_directory(
        "MAIN dataset to convert to ZIP-ready images/split + labels/split layout:",
        BASE_DIRECTORY,
    ).resolve()
    if root not in conversion_candidates_for_root(root):
        raise ValueError(
            "Selected folder is not in train/{images,labels}, val or valid/{images,labels}, "
            "test/{images,labels} working layout."
        )
    if (root / "images").exists() or (root / "labels").exists():
        raise RuntimeError(
            "Destination already contains images or labels folder. Operation halted to prevent overwrite risk."
        )
    aliases, summary = inspect_old_layout(root)
    print("\nConversion plan:")
    for split in ("train", "val", "test"):
        ni, nl, nb, ne = summary[split]
        print(f"  {aliases[split]} -> {split}: images={ni}, labels={nl}, boxes={nb}, negatives={ne}")
    print("\nOld: train/images + train/labels")
    print("New: images/train + labels/train")
    print("WARNING: This is the final packaging step; do not run operations 1-3 afterwards.")
    if not ask_confirm("Convert main dataset structure now?", False):
        print("No changes made.")
        return

    data, names, yaml_path = load_yaml(root, required=True)
    should_save, yaml_backup_name = ask_save_backup(
        "layout conversion",
        "of existing data.yaml file",
        f"{root.name}_layout_yaml_backup",
    )
    if should_save:
        yaml_backup = root.parent / yaml_backup_name
        if yaml_backup.exists() and not ask_confirm(f"{yaml_backup} already exists. Overwrite?", False):
            print("data.yaml save operation cancelled.")
            return
        with ZipFile(yaml_backup, "w", ZIP_DEFLATED) as archive:
            archive.write(yaml_path, yaml_path.relative_to(root))
        print("data.yaml save file:", yaml_backup)
    else:
        yaml_backup = None
        print("data.yaml save file was not created.")

    temp = root / f"_layout_conversion_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    if temp.exists():
        raise FileExistsError(f"Temporary folder already exists: {temp}")
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
        print(f"Conversion interrupted. Recovery files are located here: {temp}")
        raise

    output = dict(data)
    output["train"] = "images/train"
    output["val"] = "images/val"
    output["test"] = "images/test"
    output.pop("path", None)
    write_yaml(yaml_path, output, names)
    # write_yaml writes working layout; finalize package paths one last time.
    output = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    output["train"], output["val"], output["test"] = (
        "images/train", "images/val", "images/test"
    )
    yaml_path.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")
    validate_final_layout(root)
    print("Main dataset converted to ZIP-ready layout.")


def convert_to_zip_layout():
    try:
        return _impl_convert_to_zip_layout()
    except BackMenu:
        return None


def final_layout_candidates():
    """Candidates in images/{train,val,test} + labels/{train,val,test} layout."""
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
            "Selected folder is not in ZIP-ready images/{train,val,test} + "
            "labels/{train,val,test} structure."
        )
    validate_final_layout(root)
    zip_directory = choose_directory("Directory where ZIP file will be saved:", root.parent).resolve()
    output = zip_directory / f"{root.name}.zip"
    temporary = zip_directory / f".{root.name}.zip.part"
    if output.exists() and not ask_confirm(
        f"{output.name} already exists. Replace with verified new ZIP?", False
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
        # ZIP64 supports large datasets. Low compression level compresses txt/yaml files
        # without wasting unnecessary CPU on JPG/PNG.
        with ZipFile(
            temporary, "w", ZIP_DEFLATED, allowZip64=True, compresslevel=1
        ) as archive:
            # Empty split folders should also appear in archive.
            for kind in ("images", "labels"):
                for split in ("train", "val", "test"):
                    archive.writestr(f"{kind}/{split}/", "")
            processed_bytes = 0
            last_percentage = -1
            show_progress(
                "Creating ZIP", 0, max(1, total_bytes),
                f"0/{len(files)} files | 0.00/{total_bytes / (1024**3):.2f} GB",
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
                        "Creating ZIP", progress_current, max(1, progress_total),
                        f"{index}/{len(files)} files | "
                        f"{processed_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GB",
                        finish=index == len(files),
                    )
        # First close temporary ZIP completely; then rename to original in a single step.
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            print(f"Incomplete ZIP did not replace the original file: {temporary}")
        raise

    print("Checking ZIP integrity...")
    with ZipFile(output, "r") as archive:
        bad = archive.testzip()
        required = {"data.yaml", "images/train/", "images/val/", "images/test/",
                    "labels/train/", "labels/val/", "labels/test/"}
        missing = required - set(archive.namelist())
    if bad or missing:
        raise RuntimeError(f"ZIP validation failed; corrupted={bad}, missing={sorted(missing)}")
    print(f"ZIP created and verified: {output}")


def create_dataset_zip():
    try:
        return _impl_create_dataset_zip()
    except BackMenu:
        return None



def validate_flat_dataset(root, interactive=True, require_yaml=False):
    """Validate a flat images/labels directory without creating train/val/test."""
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"images and labels folders required for flat dataset: {root}"
        )

    _, names, _ = load_yaml(root, required=require_yaml)
    known = set(names) if names else None

    images = image_files(images_dir)
    labels = sorted(labels_dir.glob("*.txt"))
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    errors = []
    errors += [f"{root.name}: missing label for image {s}" for s in image_stems - label_stems]
    errors += [f"{root.name}: missing image for label {s}.txt" for s in label_stems - image_stems]

    total_boxes = total_empty = 0
    class_boxes = Counter()

    show_progress("Validating flat dataset", 0, len(labels), f"0/{len(labels)} labels")
    for index, label in enumerate(labels, 1):
        try:
            lines, counts = parse_label(label, known)
            total_boxes += len(lines)
            total_empty += not lines
            class_boxes.update(counts)
        except ValueError as exc:
            errors.append(str(exc))

        if index == len(labels) or index % max(1, len(labels) // 100) == 0:
            show_progress(
                "Validating flat dataset",
                index,
                len(labels),
                f"{index}/{len(labels)} labels",
                finish=index == len(labels),
            )

    print(f"\nValidation: {root}")
    print(
        f"  images: {len(images)}, labels: {len(labels)}, "
        f"boxes={total_boxes}, negatives={total_empty}"
    )
    if names:
        for cid in sorted(names):
            print(f"  class {cid}:{names[cid]} boxes={class_boxes[cid]}")

    if errors or len(images) != len(labels):
        raise RuntimeError(
            f"Flat dataset validation failed ({len(errors)} errors):\n"
            + "\n".join(errors[:30])
        )

    print("  RESULT: image-label pairs and YOLO labels are valid.")
    return {"flat": (len(images), len(labels), total_boxes, total_empty)}


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
    show_progress("Validating dataset", 0, len(all_labels), f"0/{len(all_labels)} labels")
    for split, paths in split_dirs(root, create=True).items():
        images, labels = image_files(paths["images"]), sorted(paths["labels"].glob("*.txt"))
        i_stems, l_stems = {p.stem for p in images}, {p.stem for p in labels}
        errors += [f"{root.name}/{split}: missing label for image {s}" for s in i_stems-l_stems]
        errors += [f"{root.name}/{split}: missing image for label {s}.txt" for s in l_stems-i_stems]
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
                    "Validating dataset", processed_labels, len(all_labels),
                    f"{processed_labels}/{len(all_labels)} labels",
                    finish=processed_labels == len(all_labels),
                )
        summaries[split] = len(images), len(labels), boxes, empty
        total_images += len(images); total_labels += len(labels)
        total_boxes += boxes; total_empty += empty
    print(f"\nValidation: {root}")
    for split, (ni, nl, nb, ne) in summaries.items():
        print(f"  {split:5s}: images={ni}, labels={nl}, boxes={nb}, negatives={ne} "
              f"[{'OK' if ni == nl else 'ERROR'}]")
    if names:
        for cid in sorted(names):
            print(f"  class {cid}:{names[cid]} boxes={class_boxes[cid]}")
    print(f"  TOTAL: images={total_images}, labels={total_labels}, "
          f"boxes={total_boxes}, negatives={total_empty}")
    if errors or total_images != total_labels:
        raise RuntimeError(f"Validation failed ({len(errors)} errors):\n" + "\n".join(errors[:30]))
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
        "images folder containing photos for which empty labels will be created:",
        BASE_DIRECTORY,
    ).resolve()
    images = image_files(images_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in selected directory: {images_dir}")
    stems = [image.stem for image in images]
    duplicate_stems = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicate_stems:
        raise RuntimeError(
            "Multiple images found with the same base name; a single label name cannot belong to two images: "
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
    print(f"  Images={len(images)}, existing matching labels={overwrite_count}")
    if existing_items:
        if not ask_confirm(
            "Labels folder is not empty. Label files with the same name as each image "
            "will be written with empty content. Do you want to overwrite?",
            False,
        ):
            print("Empty label creation cancelled.")
            return
    elif not ask_confirm(
        "Should an empty .txt label file with the same name be created for each image?",
        False,
    ):
        print("Empty label creation cancelled.")
        return

    if existing_labels:
        should_save, backup_name = ask_save_backup(
            "empty label creation",
            "of existing label files",
            "labels_empty_backup",
        )
        if should_save:
            backup = images_dir.parent / backup_name
            if backup.exists() and not ask_confirm(f"{backup} already exists. Overwrite?", False):
                print("Label save operation cancelled.")
                return
            with ZipFile(backup, "w", ZIP_DEFLATED) as archive:
                show_progress("Saving existing labels", 0, len(existing_labels), f"0/{len(existing_labels)}")
                for index, label in enumerate(existing_labels, 1):
                    archive.write(label, f"labels/{label.name}")
                    if index == len(existing_labels) or index % max(1, len(existing_labels) // 100) == 0:
                        show_progress(
                            "Saving existing labels", index, len(existing_labels),
                            f"{index}/{len(existing_labels)}", finish=index == len(existing_labels),
                        )
            print("Existing labels save file:", backup)
        else:
            print("Existing labels save file was not created.")

    labels_dir.mkdir(parents=True, exist_ok=True)
    show_progress("Creating empty labels", 0, len(images), f"0/{len(images)} labels")
    for index, image in enumerate(images, 1):
        (labels_dir / f"{image.stem}.txt").write_text("", encoding="utf-8")
        if index == len(images) or index % max(1, len(images) // 100) == 0:
            show_progress(
                "Creating empty labels", index, len(images), f"{index}/{len(images)} labels",
                finish=index == len(images),
            )
    invalid = [label for label in matching_existing if not label.is_file() or label.stat().st_size != 0]
    if invalid:
        raise RuntimeError("Empty label validation failed: " + ", ".join(map(str, invalid[:20])))
    print(f"Completed: empty labels created for {len(images)} images: {labels_dir}")


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
                        "Action to perform:",
                        [questionary.Choice("(1) Select box / Draw YOLO box", "annotate"),
                         questionary.Choice("(2) Create empty/negative labels for photos", "empty_labels"),
                         questionary.Choice("(3) Filter/reduce classes", "filter"),
                         questionary.Choice("(4) Merge datasets into main dataset", "merge"),
                         questionary.Choice("(5) Redistribute with train/val/test ratios", "split"),
                         questionary.Choice("(6) Convert main dataset to images/split + labels/split layout", "convert"),
                         questionary.Choice("(7) Validate datasets only", "validate"),
                         questionary.Choice("(8) Pack main dataset into a ZIP file", "zip"),
                         questionary.Choice("(0) Exit", "exit")],
                    )

                    if action == "exit":
                        print("Exited.")
                        return

                    {"annotate": annotate_images_with_boxes,
                     "empty_labels": create_empty_labels_for_images,
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
        print("\nNo previous menu to go back to.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERROR [{type(exc).__name__}]: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        raise SystemExit(1)