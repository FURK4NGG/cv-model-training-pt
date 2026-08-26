"""Etkilesimli YOLO dataset yonetim araci.

Duzen: dataset/{train,val,test}/{images,labels} ve data.yaml
Kurulum: py -m pip install questionary PyYAML
Calistirma: py merge_yolo_datasets.py
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
    raise SystemExit("Kurulum: py -m pip install questionary PyYAML") from exc

BASE_DIRECTORY = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DATA_YAML_NAME = "data.yaml"
DEFAULT_RATIOS = (80.0, 10.0, 10.0)
SEED = 42
CREATE_BACKUP = True


def ask_save_backup(purpose, description, default_name):
    """Ask whether a ZIP backup is wanted BEFORE asking for its filename."""
    should_save = ask_confirm(
        f"{description} icin save dosyasi olusturmak ister misiniz?",
        True,
    )
    if not should_save:
        return False, None

    filename = ask_text(
        f"{description} icin save dosyasinin adi (ornek: {default_name}.zip):",
        f"{default_name}.zip",
    )
    if not filename:
        raise ValueError("Save dosyasi adi bos olamaz.")
    if not filename.lower().endswith(".zip"):
        filename += ".zip"
    filename = Path(filename).name
    if filename in {".", ".."} or any(char in filename for char in '<>:"/\\|?*'):
        raise ValueError("Gecersiz save dosyasi adi.")
    return True, filename


def create_selected_backup(root, purpose, description, default_name, files=None, include_yaml=True):
    """Create a user-approved ZIP backup. Returns the ZIP path or None."""
    should_save, filename = ask_save_backup(purpose, description, default_name)
    if not should_save:
        print("Save dosyasi olusturulmadi.")
        return None

    target = root.parent / filename
    if target.exists():
        if not ask_confirm(f"{target} zaten var. Uzerine yazilsin mi?", False):
            print("Save islemi iptal edildi.")
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
    print("Save dosyasi:", target)
    return target

# Checkbox secili klasorlerinin basinda tam olarak "*" gorunsun.
questionary_common.INDICATOR_SELECTED = "*"
questionary_common.INDICATOR_UNSELECTED = " "

# Program baslangicinda configure_platform() tarafindan bir kez belirlenir.
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
    values.append(questionary.Choice("(Geri)", "__BACK__"))
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
    # Choice basliklarinda yazdigimiz numaralarin questionary tarafindan ikinci
    # kez "1)" seklinde eklenmesini engeller.
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
        instruction="(Yon tuslari: gezin, Space: sec/kaldir, Enter: onayla)",
    ).ask()
    if value is None:
        raise KeyboardInterrupt
    if "__BACK__" in value:
        raise BackMenu
    return value


def normalized_name(value):
    return " ".join(str(value).strip().casefold().split())


def show_progress(title, current, total, detail="", finish=False):
    """Windows ve Linux terminalinde tek satirlik yuzde ilerlemesi."""
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
    """Isletim sistemi secimini program boyunca kullanmak uzere bir kez alir."""
    global SELECTED_OS, SELECTED_LINUX_DISTRO

    while True:
        SELECTED_OS = ask_select(
            "Hangi isletim sisteminde calistiriyorsunuz?",
            [questionary.Choice("(1) Windows", "windows"),
             questionary.Choice("(2) Linux", "linux")],
        )

        if SELECTED_OS == "linux":
            try:
                SELECTED_LINUX_DISTRO = ask_select(
                    "Linux dagitiminiz:",
                    [questionary.Choice("(1) Arch / Arch tabanli", "arch"),
                     questionary.Choice("(2) Debian / Ubuntu tabanli", "debian"),
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
            f"UYARI: Python ortami {detected!r} olarak gorunuyor fakat "
            f"{SELECTED_OS!r} sectiniz. Isletim sistemine ozel islemler seciminize gore calisacak."
        )


def split_dirs(root, create=False):
    result = {}
    for split in SPLITS:
        split_name = split
        # Eski Roboflow klasorlerini okuyabil; yeni olusturulan her sey "val" olur.
        if split == "val" and not (root / "val").exists() and (root / "valid").exists():
            split_name = "valid"
        images, labels = root / split_name / "images", root / split_name / "labels"
        if create:
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
        result[split] = {"images": images, "labels": labels}
    return result


def looks_like_dataset(root):
    root = Path(root)
    if not root.is_dir():
        return False

    # Layout A: train/images, train/labels, ...
    for split in (*SPLITS, "valid"):
        if (
            (root / split / "images").is_dir()
            or (root / split / "labels").is_dir()
        ):
            return True

    # Layout B: images/train, images/val, images/test,
    #            labels/train, labels/val, labels/test
    for split in (*SPLITS, "valid"):
        if (
            (root / "images" / split).is_dir()
            or (root / "labels" / split).is_dir()
        ):
            return True

    return False


def validation_split_dirs(root):
    """Return existing image/label directories without creating anything.

    Supports both YOLO layouts:
      1. root/train/images + root/train/labels
      2. root/images/train + root/labels/train
    """
    root = Path(root)
    result = {}

    for split in SPLITS:
        # Prefer the layout that actually exists.
        direct_images = root / split / "images"
        direct_labels = root / split / "labels"

        split_images = root / "images" / split
        split_labels = root / "labels" / split

        if direct_images.is_dir() or direct_labels.is_dir():
            result[split] = {
                "images": direct_images,
                "labels": direct_labels,
                "layout": "split/images + split/labels",
            }
        elif split_images.is_dir() or split_labels.is_dir():
            result[split] = {
                "images": split_images,
                "labels": split_labels,
                "layout": "images/split + labels/split",
            }
        else:
            result[split] = {
                "images": direct_images,
                "labels": direct_labels,
                "layout": "missing",
            }

    return result


def dataset_folders():
    return sorted(
        (p for p in BASE_DIRECTORY.iterdir() if looks_like_dataset(p)),
        key=lambda p: p.name.casefold(),
    )


def choose_one_dataset(message, exclude=None):
    excluded = {p.resolve() for p in (exclude or set())}
    folder = choose_directory(message, BASE_DIRECTORY).resolve()
    if folder in excluded:
        raise ValueError("Bu klasor bu islemde tekrar veya hedef olarak secilemez.")

    # Accept a standard YOLO dataset root:
    #   dataset/
    #     train/images + train/labels
    #     val/images   + val/labels
    #     test/images  + test/labels
    #
    # Also accept a split directory or an images directory. The validator
    # should validate the selected dataset itself, not require the current
    # working directory to be a dataset.
    if looks_like_dataset(folder):
        return folder

    # If the selected folder is a dataset root whose splits are valid,
    # looks_like_dataset() may reject it because its implementation expects
    # a different layout. Detect the actual filesystem structure directly.
    split_found = False
    for split in SPLITS:
        split_dir = folder / split
        if not split_dir.is_dir():
            continue
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if images_dir.is_dir() or labels_dir.is_dir():
            split_found = True
            break

    if split_found:
        return folder

    # Support the alternate layout labels/<split> if present.
    for split in SPLITS:
        images_dir = folder / split / "images"
        labels_dir = folder / "labels" / split
        if images_dir.is_dir() or labels_dir.is_dir():
            return folder

    raise ValueError(
        "Secilen klasor taninan bir YOLO dataset yapisi degil. "
        "Dataset kokunde train/val/test altinda images ve/veya labels "
        f"klasorleri bulunamadi: {folder}"
    )


def choose_destination_folder(message, exclude=None):
    """Bos veya mevcut bir hedef klasoru terminal gezginiyle sectirir."""
    excluded = {p.resolve() for p in (exclude or set())}
    folder = choose_directory(message, BASE_DIRECTORY).resolve()
    if folder in excluded:
        raise ValueError("Kaynak olarak secilen klasor hedef olarak kullanilamaz.")
    return folder


def choose_many_datasets(message, exclude=None, allow_empty=False):
    """Farkli dizinlerden bir veya daha fazla dataset klasoru sectirir."""
    excluded = {p.resolve() for p in (exclude or set())}
    selected = []
    if allow_empty and not ask_confirm(f"{message} Herhangi bir klasor secilecek mi?", False):
        return selected
    while True:
        folder = choose_one_dataset(message, excluded | {p.resolve() for p in selected})
        selected.append(folder)
        print(f"Secildi ({len(selected)}): {folder}")
        if not ask_confirm("Bu gruba baska dataset klasoru eklensin mi?", False):
            return selected


def choose_ordered_datasets(message, available=None):
    """Farkli dizinlerden kaynaklari secim sirasini koruyarak sectirir."""
    print(
        "Secilen klasorlerin data.yaml dosyalarindaki class'lar ana data.yaml'da kullanilir; "
        "klasor adlari class adi olarak kullanilmaz."
    )
    print("Secme sirasi ana data.yaml class sirasini belirler.")
    return choose_many_datasets(message, allow_empty=False)


def choose_directory(message, start=None):
    """Terminalde klasorler arasinda gezerek bir hedef dizin sectirir."""
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
            ("class:instruction", f"Bulunulan dizin: {current}\n"),
            ("class:instruction",
             "Yukari/asagi: gezin | Sag/Space: klasore gir | Sol/Esc: ust dizin | "
             "Enter: bu dizini sec | Ctrl+C: iptal\n\n"),
        ]
        if not folders:
            tokens.append(("class:instruction", "  (Alt klasor yok veya okuma izni yok)\n"))
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
            f"Secilen images dizininin yaninda labels klasoru bulunamadi: {images_root}"
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
                    f"{split} icin labels klasoru bulunamadi: {labels_root}"
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
                raise RuntimeError("Windows resim acma islemi Windows disinda calistirilamaz.")
            os.startfile(str(path))
        elif SELECTED_OS == "linux":
            if shutil.which("xdg-open") is None:
                install_commands = {
                    "arch": "sudo pacman -S xdg-utils",
                    "debian": "sudo apt install xdg-utils",
                    "fedora": "sudo dnf install xdg-utils",
                }
                command = install_commands.get(SELECTED_LINUX_DISTRO, "xdg-utils paketini kurun")
                raise RuntimeError(f"xdg-open bulunamadi. Kurulum: {command}")
            subprocess.Popen(["xdg-open", str(path)])
        else:
            raise RuntimeError("Isletim sistemi secimi yapilmadi.")
    except Exception as exc:
        print(f"Resim acilamadi: {exc}")


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
            raise ValueError(f"Gecersiz YOLO satiri: {label}:{line_no}\n{raw}") from exc
        if class_id < 0:
            raise ValueError(f"Negatif class ID: {label}:{line_no}")
        if known_ids is not None and class_id not in known_ids:
            raise ValueError(f"data.yaml'da olmayan class ID {class_id}: {label}:{line_no}")
        is_box = len(parts) == 5
        is_polygon = len(coords) >= 6 and len(coords) % 2 == 0
        if not (is_box or is_polygon):
            raise ValueError(f"Kutu/polygon formati gecersiz: {label}:{line_no}")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
            raise ValueError(f"Koordinat 0-1 disinda: {label}:{line_no}")
        if is_box and (coords[2] <= 0 or coords[3] <= 0):
            raise ValueError(f"Gecersiz kutu boyutu: {label}:{line_no}")
        lines.append((class_id, parts[1:]))
        counts[class_id] += 1
    return lines, counts


def load_yaml(root, required=True):
    path = root / DATA_YAML_NAME
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"data.yaml bulunamadi: {path}")
        return {}, {}, path
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    names = data.get("names")
    if isinstance(names, list):
        mapping = {i: str(v).strip() for i, v in enumerate(names)}
    elif isinstance(names, dict):
        mapping = {int(i): str(v).strip() for i, v in names.items()}
    else:
        raise ValueError(f"data.yaml icinde gecerli names yok: {path}")
    duplicate = [n for n, c in Counter(normalized_name(v) for v in mapping.values()).items() if c > 1]
    if not mapping or any(not v for v in mapping.values()) or duplicate:
        raise ValueError(f"Bos veya tekrarli class adi: {path}; tekrar={duplicate}")
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
        raise FileNotFoundError(f"data.yaml bulunamadi: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    raw_names = data.get("names")

    if isinstance(raw_names, list):
        names = {i: str(value).strip() for i, value in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(i): str(value).strip() for i, value in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"data.yaml icindeki class ID'leri gecersiz: {path}"
            ) from exc
    else:
        raise ValueError(f"data.yaml icinde gecerli names yok: {path}")

    if not names:
        raise ValueError(f"data.yaml icinde class bulunamadi: {path}")

    invalid_ids = [cid for cid in names if cid < 0]
    empty_names = [cid for cid, name in names.items() if not name]
    if invalid_ids:
        raise ValueError(
            f"data.yaml icinde negatif class ID var: {invalid_ids}"
        )
    if empty_names:
        raise ValueError(
            f"data.yaml icinde bos class adi var: {empty_names}"
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
        'SPLITS=("train", "val", "test"). Her dataset klasorunde '
        "split/images ve split/labels duzenini dogruluyor musunuz?", True
    ):
        raise SystemExit("Klasor yapisini duzeltip yeniden calistirin.")


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
            raise RuntimeError(f"Ayni kok adli birden fazla resim: {stem} -> {files}")
        for image in list(images):
            label = paths["labels"] / f"{image.stem}.txt"
            if label.is_file():
                continue
            action = ask_select(
                f"{image.name} isimli dosyaya ait label bulunamadi ({root.name}/{split}).",
                [questionary.Choice("(1) Resmi ac", "open"),
                 questionary.Choice("(2) Devam et / simdilik atla", "continue"),
                 questionary.Choice("(3) Programi sonlandir", "exit")],
            )
            if action == "exit":
                raise SystemExit("Eksik label nedeniyle sonlandirildi.")
            if action == "continue":
                continue
            open_path(image)
            resolution = ask_select(
                f"{image.name} icin ne yapilsin?",
                [questionary.Choice("(1) Resmi sil", "delete"),
                 questionary.Choice("(2) Bos label olustur", "empty"),
                 questionary.Choice("(3) Hicbir sey yapma ve bitir", "exit")],
            )
            if resolution == "delete":
                image.unlink()
            elif resolution == "empty":
                label.touch()
            else:
                raise SystemExit("Kullanici istegiyle sonlandirildi.")
        stems = {p.stem for p in image_files(paths["images"])}
        orphans = [p for p in paths["labels"].glob("*.txt") if p.stem not in stems]
        if orphans:
            raise RuntimeError("Resmi bulunmayan label dosyalari:\n" +
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
    show_progress("Dataset tarama", 0, len(items), f"0/{len(items)} resim")
    for index, (split, paths, image) in enumerate(items, 1):
            label = paths["labels"] / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Label bulunamadi: {label}")
            _, counts = parse_label(label, known)
            result.append(Pair(image, label, split, frozenset(counts), counts))
            if index == len(items) or index % max(1, len(items) // 100) == 0:
                show_progress(
                    "Dataset tarama", index, len(items), f"{index}/{len(items)} resim",
                    finish=index == len(items),
                )
    return result


def backup_metadata(root, purpose):
    if not CREATE_BACKUP:
        return None

    default_name = f"{root.name}_{purpose}_backup"
    should_save, filename = ask_save_backup(
        purpose,
        f"{root.name} datasetindeki mevcut dosyalarin yedegi",
        default_name,
    )
    if not should_save:
        print("Save dosyasi olusturulmadi.")
        return None

    target = root.parent / filename
    if target.exists():
        if not ask_confirm(f"{target} zaten var. Uzerine yazilsin mi?", False):
            print("Save islemi iptal edildi.")
            return None

    yml = root / DATA_YAML_NAME
    files = ([yml] if yml.is_file() else []) + [
        label
        for paths in split_dirs(root).values()
        if paths["labels"].is_dir()
        for label in sorted(paths["labels"].glob("*.txt"))
    ]

    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        show_progress("Save olusturma", 0, len(files), f"0/{len(files)} dosya")
        for index, file in enumerate(files, 1):
            archive.write(file, file.relative_to(root))
            if index == len(files) or index % max(1, len(files) // 100) == 0:
                show_progress(
                    "Save olusturma", index, len(files), f"{index}/{len(files)} dosya",
                    finish=index == len(files),
                )
    print("Save dosyasi:", target)
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
            raise ValueError(f"Class ID icin hedef esleme yok: {old_id}")
        output.append(" ".join([str(id_map[old_id]), *coords]))
    return "\n".join(output) + ("\n" if output else "")


def merge_classes_within_dataset(root, data, names, yaml_path):
    selected = set(ask_checkbox(
        "Tek class'ta birlestirilecek class'lari secin:",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(names)],
    ))
    if len(selected) < 2:
        raise ValueError("Birlestirmek icin en az iki class secilmelidir.")
    target_id = ask_select(
        "Secilen class'lar hangi hedef class adinda birlessin?",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(selected)],
    )

    # Hedef class eski konumunda kalir; birlestirilen diger class'lar kaldirilir.
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
    show_progress("Class birlestirme taramasi", 0, len(pairs), f"0/{len(pairs)} label")
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
                "Class birlestirme taramasi", index, len(pairs), f"{index}/{len(pairs)} label",
                finish=index == len(pairs),
            )

    print("\nClass birlestirme plani:")
    for cid in sorted(selected):
        print(
            f"  {cid}:{names[cid]} -> "
            f"{retained_to_new[target_id]}:{names[target_id]} "
            f"(kutu={selected_box_counts[cid]})"
        )
    print("\nYeni data.yaml class sirasi:")
    for cid in sorted(new_names):
        print(f"  {cid}: {new_names[cid]}")
    print(
        f"Degisecek label={changed_labels}, hedef class'a cevrilecek kutu={merged_boxes}, "
        f"bos/negatif label={empty_labels}"
    )
    if not ask_confirm("Class birlestirme planini uygula?", False):
        print("Degisiklik yapilmadi.")
        return

    backup = backup_metadata(root, "class_merge")
    if backup:
        print("Label/YAML yedegi:", backup)
    show_progress("Class birlestirme", 0, len(plans), f"0/{len(plans)} label")
    for index, (label, text) in enumerate(plans, 1):
        label.write_text(text, encoding="utf-8")
        if index == len(plans) or index % max(1, len(plans) // 100) == 0:
            show_progress(
                "Class birlestirme", index, len(plans), f"{index}/{len(plans)} label",
                finish=index == len(plans),
            )
    write_yaml(yaml_path, data, new_names)
    validate_dataset(root, interactive=False, require_yaml=True)
    print(
        f"Class birlestirme tamamlandi: {', '.join(names[c] for c in sorted(selected))} "
        f"-> {names[target_id]}"
    )


def force_all_labels_to_class_id(root, data, names, yaml_path):
    """
    Butun dolu label satirlarinin class ID'sini tek bir hedef ID'ye cevirir.

    Hedef iki sekilde secilebilir:
      1) data.yaml icindeki mevcut class'lardan biri
      2) Kullanicinin kendi belirledigi yeni bir class ID

    Yeni bir ID girildiginde kullaniciya bunun data.yaml'a da eklenip
    eklenmeyecegi ayrica sorulur. Hayir denirse mevcut data.yaml korunur.
    """

    created_new_class = False

    if names:
        target_choice = ask_select(
            "Butun dolu label satirlari hangi hedef class'a donusturulsun?",
            [
                questionary.Choice(
                    "data.yaml icindeki bir class'i sec",
                    "yaml_class",
                ),
                questionary.Choice(
                    "Yeni class ID'sini kendin belirle",
                    "new_id",
                ),
            ],
        )
    else:
        print(
            "data.yaml icinde secilebilecek class bulunamadi; "
            "yeni class ID'sini kendiniz belirlemelisiniz."
        )
        target_choice = "new_id"

    if target_choice == "yaml_class":
        new_id = ask_select(
            "data.yaml icindeki hedef class'i secin:",
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
            "Yeni class ID'sini girin (0 veya daha buyuk bir tam sayi):"
        )
        try:
            new_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(
                "Class ID, 0 veya daha buyuk bir tam sayi olmalidir."
            ) from exc

        if new_id < 0:
            raise ValueError("Class ID negatif olamaz.")

        if new_id in names:
            raise ValueError(
                f"{new_id} ID'si zaten data.yaml icinde mevcut: "
                f"{names[new_id]}. Mevcut class'i secmek icin "
                "onceki menuden 'data.yaml icindeki bir class'i sec' secenegini kullanin."
            )

        target_name = ask_text(
            "Yeni class'in adini girin (data.yaml'a yazilacaksa kullanilir):"
        )
        if not target_name:
            raise ValueError("Class adi bos olamaz.")

        # Ayni isimde baska bir class varsa yeni bir class adi olusturmak
        # data.yaml acisindan anlamsiz olur.
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
            # Ayni class adi baska bir ID'de varsa HATA VERME.
            # Kullaniciya sor ve onaylarsa eski ID'yi data.yaml'dan kaldir.
            overwrite_name = ask_confirm(
                f"'{target_name}' adi data.yaml icinde zaten "
                f"{duplicate_name} ID'sinde kullaniliyor. "
                f"{new_id} ID'sine tasinmasina ve eski {duplicate_name} ID'sinin "
                "data.yaml'dan kaldirilmasina izin veriyor musunuz?",
                False,
            )

            if not overwrite_name:
                raise ValueError(
                    f"'{target_name}' adi zaten {duplicate_name} ID'sinde kullaniliyor; "
                    "islem iptal edildi."
                )

            names = dict(names)
            del names[duplicate_name]

        created_new_class = True

    if new_id != 0:
        print(
            "UYARI: Tek class'li bir datasetin egitim ID'si normalde 0 olmalidir. "
            f"{new_id} degeri ana datasete birlestirme oncesi hazirlik icin kullanilabilir; "
            "bu ara dataseti dogrudan egitmeyin."
        )

    # Bu modun amaci bozuk/eski ID'leri onarmaktir. Bu nedenle eski ID'nin
    # data.yaml'da tanimli olmasi zorunlu tutulmaz; satir formati ve koordinatlar
    # yine eksiksiz dogrulanir.
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

    print("\nToplu class ID degistirme plani:")
    for old_id in sorted(old_id_counts):
        old_name = names.get(old_id, "<data.yaml'da tanimli degil>")
        print(
            f"  {old_id}:{old_name} -> "
            f"{new_id}:{target_name} "
            f"(kutu={old_id_counts[old_id]})"
        )

    print(
        f"Toplam kutu={box_count}, degisecek label={len(plans)}, "
        f"zaten dogru dolu label={already_correct}, "
        f"bos/negatif label={empty_count}"
    )

    if created_new_class:
        print(
            f"\nYeni class:"
            f"\n  ID   : {new_id}"
            f"\n  Ad   : {target_name}"
        )

        update_yaml = ask_confirm(
            "Bu yeni class ID'si data.yaml icine de islenmesini ister misiniz?",
            True,
        )
    else:
        # Mevcut bir data.yaml class'i secildiginde YAML zaten o class'i
        # tanimladigi icin ekstra bir class ekleme islemi gerekmez.
        update_yaml = False

    if not ask_confirm(
        "Butun dolu label class ID'leri degistirilsin mi?",
        False,
    ):
        print("Degisiklik yapilmadi.")
        return

    backup = backup_metadata(root, "class_id")
    if backup:
        print("Label/YAML yedegi:", backup)

    show_progress(
        "Label ID degistirme",
        0,
        len(plans),
        f"0/{len(plans)} label",
    )

    for index, (label, text) in enumerate(plans, 1):
        label.write_text(text, encoding="utf-8")

        if (
            index == len(plans)
            or index % max(1, len(plans) // 100) == 0
        ):
            show_progress(
                "Label ID degistirme",
                index,
                len(plans),
                f"{index}/{len(plans)} label",
                finish=index == len(plans),
            )

    if update_yaml:
        # Mevcut data.yaml'daki class'lari koru ve yeni ID'yi sona eklemek
        # yerine kullanicinin verdigi ID'ye dogrudan yerlestir.
        updated_names = dict(names)
        updated_names[new_id] = target_name

        write_yaml(
            yaml_path,
            data,
            updated_names,
        )

        print(
            f"data.yaml guncellendi: "
            f"{new_id}: {target_name}"
        )
    else:
        if created_new_class:
            print(
                "data.yaml degistirilmedi. "
                f"Yeni ID {new_id} label dosyalarina yazildi fakat "
                "data.yaml'a eklenmedi."
            )
        else:
            print(
                "Mevcut data.yaml class'i kullanildi; "
                "data.yaml'da ek degisiklik yapilmadi."
            )

    # Yeni ID data.yaml'a yazilmadiysa validate_dataset() bunu
    # bilinmeyen class ID olarak reddeder. Bu durumda yalnızca label
    # formatini ve koordinatlarini kontrol et.
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
        f"Toplu class ID degistirme tamamlandi. "
        f"Butun dolu kutular: {new_id}:{target_name}"
    )


def extract_or_delete_class_pairs(root, data, names):
    selected = set(ask_checkbox(
        "Filtrelenmesini istediğiniz class'ları seçin:",
        [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(names)],
    ))
    if not selected:
        raise ValueError("En az bir class secilmelidir.")

    policy = ask_select(
        "Nasil resimler filtrelenmeli?",
        [questionary.Choice(
            "Yeni data.yaml'da class'i bulunmayacak bir obje, modeli secilen class icin "
            "egiten resimde bulunabilir (onerilen)",
            "allow_other"),
         questionary.Choice(
            "Sadece sectigimiz class'larin box'inin bulundugu resimler kullanilsin",
            "only_effective")],
    )
    action = ask_select(
        "Yapmak istediğiniz işlem:",
        [questionary.Choice("Başka dosyaya kopyalama", "copy"),
         questionary.Choice("Silmek", "delete")],
    )

    destination = None
    if action == "copy":
        destination_parent = choose_directory(
            "Kopyanin kaydedilecegi ust dizini secin:", BASE_DIRECTORY
        )
        destination_name = ask_text(
            "Yeni hedef dataset klasorunun adi "
            "(bos birakilirsa train/val/test ve data.yaml secilen dizinin icinde "
            "kullanilir veya olusturulur):"
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
            raise ValueError("Hedef dataset kaynak datasetin icinde olamaz.")

    scope = ask_select(
        "Hangi tür için geçerli olsun:",
        [questionary.Choice("Seçilen class'lar", "selected"),
         questionary.Choice("Seçilmeyen class'lar", "unselected")],
    )
    effective_ids = selected if scope == "selected" else set(names) - selected
    if not effective_ids:
        raise ValueError("Bu secim sonucunda kullanilabilecek class kalmadi.")

    # Filtrelenmis dataset kendi icinde 0'dan baslayan ardışık ID'lere sahip olur.
    filtered_order = [cid for cid in sorted(names) if cid in effective_ids]
    filtered_id_by_old = {old_id: new_id for new_id, old_id in enumerate(filtered_order)}
    destination_data = dict(data)
    destination_names = {}
    destination_yaml = destination / DATA_YAML_NAME if destination is not None else None
    copy_id_map = dict(filtered_id_by_old)

    if action == "copy":
        if (destination / "images").is_dir() or (destination / "labels").is_dir():
            raise RuntimeError(
                "Secilen hedef images/train + labels/train ZIP duzeninde gorunuyor. "
                "Bu filtreleme islemi train/images + train/labels calisma duzenini kullanir."
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
                        f"Kopyalanacak data.yaml icinde {filtered_id}:{class_name}, hedef "
                        f"data.yaml icinde {target_id}:{output_names[target_id]} var. "
                        "Dataseti korumak icin kopyalanacak label class ID degeri "
                        f"{target_id} yapilsin mi? No secilirse islem iptal edilir.",
                        True,
                    ):
                        raise SystemExit("Class ID eslemesi reddedildi; kopyalama iptal edildi.")
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
                        f"Kopyalanacak {filtered_id}:{class_name} ID'si hedef data.yaml icinde "
                        f"{filtered_id}:{conflicting_name} tarafindan kullaniliyor. En yakin bos "
                        f"ID {target_id}. Filtrelenmis {class_name} label ID'leri "
                        f"{target_id} yapilsin mi? No secilirse islem iptal edilir.",
                        True,
                    ):
                        raise SystemExit("Bos class ID onerisi reddedildi; kopyalama iptal edildi.")
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
                    "Hedefte data.yaml yok fakat mevcut label dosyalari var. Class ID "
                    "anlamlari bilinmedigi icin guvenli kopyalama yapilamaz."
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
            raise ValueError(f"{cid}:{names[cid]} icin uygun resim bulunamadi.")
        raw = ask_text(
            f"{cid}:{names[cid]} icin kac tane filtrelensin? "
            f"(bulunan={len(candidates)}, 0=tumu):",
            "0",
        )
        try:
            amount = int(raw)
        except ValueError as exc:
            raise ValueError("Filtreleme adedi 0 veya daha buyuk bir tam sayi olmalidir.") from exc
        if amount < 0 or amount > len(candidates):
            raise ValueError(
                f"{cid}:{names[cid]} icin adet 0 ile {len(candidates)} arasinda olmalidir."
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
        raise ValueError("Secime uyan resim-label cifti bulunamadi.")
    mixed_pairs = sum(bool(present - effective_ids) for _, _, present in chosen)

    print("\nClass verisi islem plani:")
    print("  Kaynak:", root)
    print("  Islem:", "baska datasete kopyalama" if action == "copy" else "resim-label ciftlerini silme")
    if destination is not None:
        print("  Hedef:", destination)
    print("  Kapsam:", ", ".join(f"{cid}:{names[cid]}" for cid in sorted(effective_ids)))
    if action == "copy":
        print("  Hedef data.yaml class eslemeleri:")
        for old_id in filtered_order:
            filtered_id = filtered_id_by_old[old_id]
            target_id = copy_id_map[old_id]
            print(f"    {filtered_id}:{names[old_id]} -> {target_id}:{destination_names[target_id]}")
    for cid in sorted(effective_ids):
        amount_text = "tumu" if requested[cid] == 0 else str(requested[cid])
        print(f"  {cid}:{names[cid]} -> istek={amount_text}, uygun={available[cid]}")
        print(
            "    mevcut: "
            + ", ".join(f"{split}={split_available[cid][split]}" for split in SPLITS)
        )
        print(
            "    secilecek: "
            + ", ".join(f"{split}={split_requested[cid][split]}" for split in SPLITS)
        )
    print(f"  Tekrarsiz islenecek cift={len(chosen)}, karisik-class resim={mixed_pairs}")
    if len(effective_ids) > 1:
        print(
            "  NOT: Bir resimde birden fazla hedef class varsa bu resim her class'in secimine "
            "katkida bulunabilir fakat yalniz bir kez kopyalanir/silinir."
        )
    if action == "copy" and mixed_pairs:
        print(
            "  UYARI: Kapsam disi objeler resimde gorunmeye devam eder ancak kutulari yeni "
            "label'a yazilmaz; egitimde arka plan gibi degerlendirilebilirler."
        )
    if action == "delete" and mixed_pairs:
        print(
            "  UYARI: Secilen adaylarin bazilarinda kapsam disi class da var. Silme islemi "
            "bu resimleri ve label dosyalarini tamamen kaldirir."
        )
    if not ask_confirm("Bu plan uygulansin mi?", False):
        print("Degisiklik yapilmadi.")
        return

    if action == "delete":
        backup = backup_metadata(root, "class_pair_delete")
        if backup:
            print("Label/YAML yedegi:", backup)
        total = len(chosen)
        show_progress("Silme", 0, total, f"0/{total} cift")
        for index, (pair, _, _) in enumerate(chosen, 1):
            pair.image.unlink()
            pair.label.unlink()
            if index == total or index % max(1, total // 100) == 0:
                show_progress("Silme", index, total, f"{index}/{total} cift", finish=index == total)
        validate_dataset(root, False, True)
        print(f"Silme tamamlandi: {total} resim-label cifti kaldirildi.")
        return

    if destination.exists() and destination_yaml.is_file():
        backup = backup_metadata(destination, "filter_copy")
        if backup:
            print("Hedef label/YAML yedegi:", backup)
    dirs = split_dirs(destination, create=True)
    total = len(chosen)
    show_progress("Class verisi kopyalama", 0, total, f"0/{total} cift")
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
                "Class verisi kopyalama", index, total, f"{index}/{total} cift",
                finish=index == total,
            )
    write_yaml(destination_yaml, destination_data, destination_names)
    validate_dataset(destination, False, True)
    print(f"Kopyalama tamamlandi: cift={total}, kutu={copied_boxes}, hedef={destination}")



def annotate_images_with_boxes():
    """
    GUI ile YOLO bounding-box annotation yap.

    - Secilen images klasorundeki resimleri tek tek gosterir.
    - Mevcut .txt label varsa kutulari okur ve class renkleriyle gosterir.
    - Fare ile tutup surukleyerek yeni dikdortgen/kare cizilir.
    - Ustte data.yaml'dan bulunan class'lardan biri secilir.
    - Her class icin farkli bir renk kullanilir.
    - Save ile mevcut label dosyasi yeniden yazilir; label yoksa olusturulur.
    """
    if tk is None or Image is None:
        raise RuntimeError(
            "GUI icin Pillow ve Tkinter gerekli. "
            "Kurulum: python -m pip install Pillow"
        )

    selected_path = choose_directory(
        "Kutu etiketlenecek resimlerin bulundugu klasoru secin:",
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
                    "Secilen split klasorunde images klasoru veya resim bulunamadi: "
                    f"{selected_path}"
                )

    elif selected_name == "labels":
        raise ValueError(
            "Label klasoru secildi. Annotation icin images klasorunu secin."
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
                    "Secilen klasorde desteklenen resim veya YOLO split yapisi "
                    "bulunamadi: "
                    f"{selected_path}"
                )

            if len(candidates) == 1:
                split_name, images_dir = candidates[0]
            else:
                split_name = ask_select(
                    "Hangi split'i annotate etmek istiyorsunuz?",
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
            f"Resim klasoru bulunamadi: {images_dir}"
        )

    # The selected directory may be a dataset root, a split directory, an
    # images directory, or a directory containing images directly. Resolve
    # the final images/labels pair solely from its filesystem structure.
    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()

    images = image_files(images_dir)
    if not images:
        raise FileNotFoundError(
            f"Secilen dizinde desteklenen resim bulunamadi: {images_dir}"
        )

    # data.yaml'i once secilen dizinde, sonra bir ust dizinde ara.
    yaml_candidates = [
        images_dir / DATA_YAML_NAME,
        images_dir.parent / DATA_YAML_NAME,
        dataset_root / DATA_YAML_NAME,
        dataset_root.parent / DATA_YAML_NAME,
    ]
    print(
        "Annotation paths resolved:",
        f"\\n  Images: {images_dir}",
        f"\\n  Labels: {labels_dir}",
        f"\\n  Dataset root: {dataset_root}",
        f"\\n  YAML candidates: {yaml_candidates}",
    )

    yaml_path = next((p for p in yaml_candidates if p.is_file()), None)
    if yaml_path is None:
        raise FileNotFoundError(
            "Class listesi icin data.yaml bulunamadi. Aranan yerler:\n"
            + "\n".join(f"  {p}" for p in yaml_candidates)
        )

    yaml_root = yaml_path.parent
    data, names, _ = load_yaml_for_annotation(yaml_path)
    if not names:
        raise ValueError(f"data.yaml icinde class bulunamadi: {yaml_path}")

    # Label klasoru yoksa annotation sirasinda olusturulabilir.
    labels_dir.mkdir(parents=True, exist_ok=True)

    # HSV renkleri: class sayisi ne olursa olsun renkler birbirinden ayrilsin.
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
                    # Bu annotation ekraninda sadece standart YOLO bbox
                    # satirlari gosterilir; polygonlar degistirilmeden kalir.
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
        text="Class renkleri:",
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
            "Fare: kutu ciz | Mouse ile mevcut kutuya tikla: sec | "
            "Delete: secili kutuyu sil | ←/→: resim | Ctrl+S: kaydet | "
            "Enter: kaydet ve sonraki | Esc: cik"
        )
    )
    ttk.Label(
        bottom,
        textvariable=hint_var,
        foreground="#666666",
    ).pack(fill="x", pady=(0, 8))

    button_row = ttk.Frame(bottom)
    button_row.pack(fill="x")

    previous_button = ttk.Button(button_row, text="← Onceki")
    previous_button.pack(side="left", fill="x", expand=True, padx=(0, 4))

    save_button = ttk.Button(button_row, text="Kaydet")
    save_button.pack(side="left", fill="x", expand=True, padx=4)

    next_button = ttk.Button(button_row, text="Sonraki →")
    next_button.pack(side="left", fill="x", expand=True, padx=4)

    save_next_button = ttk.Button(button_row, text="Kaydet ve Sonraki")
    save_next_button.pack(side="left", fill="x", expand=True, padx=4)

    finish_button = ttk.Button(button_row, text="Cik")
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
                f"data.yaml icinde secilen class ID bulunamadi: {class_id}"
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
        dirty_var.set("* Kaydedilmemis degisiklik" if state["dirty"] else "")

    def canvas_to_image(x, y):
        if not state["display_size"] or not state["display_origin"]:
            return None
        ox, oy = state["display_origin"]
        dw, dh = state["display_size"]
        if dw <= 0 or dh <= 0:
            return None

        # Canvas koordinatini resmin 0..1 normalize edilmis koordinatina cevir.
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

        # Once kutular ve yeni kutular class'a gore farkli renkte cizilir.
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

            # Class yazisini kutunun sol ustune koy.
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

            # Okunurluk icin yazi arkasina ayni renk bir dikdortgen.
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

            # Canvas boyutunun hazir olmasi icin once update_idletasks.
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
                f"Box: {len(state['boxes'])}"
            )
            status_var.set(
                f"Label: {label if label.is_file() else 'yok (Kaydet ile olusturulacak)'}"
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
                text=f"Resim acilamadi:\n{image}\n\n{exc}",
            )
            status_var.set(f"HATA: {exc}")

    def save_current():
        image = images[state["index"]]
        label = find_label_for_image(image)

        # Sadece standart YOLO bbox'lari yaziyoruz.
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
        status_var.set(f"Kaydedildi: {label}")
        image_info_var.set(
            f"{state['index'] + 1}/{len(images)}  |  "
            f"{image.name}  |  {state['original_size'][0]}x{state['original_size'][1]}  |  "
            f"Box: {len(state['boxes'])}"
        )
        update_dirty()

    def ensure_saved_before_navigation():
        if not state["dirty"]:
            return True

        answer = questionary.confirm(
            "Mevcut resimde kaydedilmemis kutular var. Kaydetmeden gecilsin mi?",
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

        # Mevcut bir kutuya tiklanmissa onu sec.
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
                "Mevcut kutu secildi. Silmek icin Delete, yeniden class vermek icin "
                "ustteki class'i secip R tusuna basin."
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

        # Sadece resim alaninda ciz.
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

        # Tiklama kadar kucuk kutular kazara olusmasin.
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
            f"Yeni box eklendi: {class_id}: {names[class_id]}. "
            "Kaydetmek icin Ctrl+S veya Kaydet butonunu kullanin."
        )
        draw_all()

    def delete_selected_box():
        index = state["selected_box"]
        if index is None or not (0 <= index < len(state["boxes"])):
            status_var.set("Silinecek secili box yok.")
            return
        deleted = state["boxes"].pop(index)
        state["selected_box"] = None
        state["dirty"] = True
        status_var.set(
            f"Box silindi: {deleted['class_id']}: {names[deleted['class_id']]}. "
            "Degisiklik henuz kaydedilmedi."
        )
        draw_all()

    def reclass_selected_box():
        index = state["selected_box"]
        if index is None or not (0 <= index < len(state["boxes"])):
            status_var.set("Class'i degistirilecek secili box yok.")
            return
        old_id = state["boxes"][index]["class_id"]
        new_id = selected_class_id()
        state["boxes"][index]["class_id"] = new_id
        state["dirty"] = True
        status_var.set(
            f"Box class'i degistirildi: {old_id}:{names[old_id]} -> "
            f"{new_id}:{names[new_id]}. Kaydedilmedi."
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
            f"Yeni cizilecek box class'i: {selected_class_id()}: "
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
            # Enter: kaydet ve sonraki resme gec.
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
        # Pencere boyutu degistiginde resmi ve kutulari yeni alana sigdir.
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

    # Resize'i her pikselde tekrar resim acacak kadar agirlastirmamak icin
    # Tkinter'in after mekanizmasiyla gecikmeli calistir.
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

    # Pencere kapatildiysa kaydedilmemis degisiklikleri sessizce silmeyelim.
    if state["dirty"]:
        # root.quit() sonrasi burada tekrar GUI soru sormak yerine son resmi kaydet.
        # Kullanici Cik butonunu kullandiginda da kaydedilmemis kutular kaybolmasin.
        save_current()

    print(
        f"Annotation ekrani kapatildi. Islenen resim klasoru: {images_dir}\n"
        f"Label klasoru: {labels_dir}\n"
        f"Class kaynagi: {yaml_path}"
    )

def _impl_filter_classes():
    filter_action = ask_select(
        "Class islemi:",
        [questionary.Choice("(1) Class'lari tut veya kaldir", "filter"),
         questionary.Choice("(2) Birden fazla class'i tek class'ta birlestir", "merge_classes"),
         questionary.Choice("(3) Butun label class ID'lerini tek bir degere cevir", "force_id")],
    )
    root = choose_one_dataset("Class'lari filtrelenecek dataset:")
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
        raise ValueError("Yeni klasor adi bos olamaz ve dizin yolu iceremez.")
    if any(char in name for char in '<>:"/\\|?*') or name.endswith((" ", ".")):
        raise ValueError("Yeni klasor adi Windows/Linux icin gecersiz karakter iceriyor.")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                *(f"lpt{i}" for i in range(1, 10))}
    if name.casefold() in reserved:
        raise ValueError("Bu klasor adi Windows tarafindan ayrilmistir.")
    return name



def is_flat_images_labels_dataset(root):
    """Kaynakta train/val/test yok, sadece images + labels varsa True."""
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
    Flat images+labels kaynagi birlestirilirken hedef yapisini sec:
      1) train/val/test/images + labels
      2) images + labels
      3) secilen hedef dizine dogrudan kopyala
    """
    return ask_select(
        "Hedefte train/val/test klasorleri yok. Flat kaynak nasil yerlestirilsin?",
        [
            questionary.Choice(
                "train/val/test klasorlerini olustur; dosyalari train'e koy",
                "split",
            ),
            questionary.Choice(
                "Hedef icinde images ve labels klasorleri olustur",
                "flat",
            ),
            questionary.Choice(
                "Secilen hedef dizine dogrudan yapistir",
                "direct",
            ),
        ],
    )



def choose_merge_sources(message):
    """
    Merge icin kaynak olarak hem dataset kokunu hem de tek bir split klasorunu
    (train/val/valid/test) secmeye izin verir.

    ONEMLI:
      dataset/train veya dataset/valid gibi bir klasor secildiginde,
      o klasorun icinde images + labels bulunmasi onu SPLIT olarak kabul ettirir.
      Once looks_like_dataset() kontrolu yapilmaz; aksi halde train/val/valid/test
      klasoru yanlislikla dataset koku sanilip split_dirs(..., create=True)
      tarafindan icine yeni train/val/test klasorleri olusturulabiliyordu.
    """
    selected = []
    while True:
        selected_path = choose_directory(message, BASE_DIRECTORY).resolve()

        source_root = None
        selected_split = None
        name = selected_path.name.casefold()

        # 1) ONCE tek split kontrolu.
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

        # 4) Ancak bundan sonra gercek dataset kokunu kontrol et.
        elif looks_like_dataset(selected_path):
            source_root = selected_path

        else:
            raise ValueError(
                "Secilen klasor dataset koku, dataset/train|val|valid|test, "
                "images veya images/train|val|valid|test yapisinda olmali."
            )

        entry = (source_root, selected_split)
        if entry in selected:
            raise ValueError("Bu kaynak zaten secildi.")

        selected.append(entry)

        if selected_split is None:
            split_text = "tum splitler"
        else:
            split_text = f"sadece {selected_split}"

        print(f"Secildi ({len(selected)}): {source_root} -> {split_text}")

        if not ask_confirm("Bu gruba baska kaynak klasoru eklensin mi?", False):
            return selected



def manifest_selected_split_without_yaml(root, split):
    """data.yaml olmadan tek bir splitin resim-label ciftlerini tara."""
    dirs = split_dirs(root, create=False)
    paths = dirs[split]
    if not paths["images"].is_dir() or not paths["labels"].is_dir():
        raise FileNotFoundError(
            f"{root} icinde {split}/images ve {split}/labels bulunamadi."
        )

    images = image_files(paths["images"])
    show_progress(
        f"{root.name}/{split} tarama",
        0,
        len(images),
        f"0/{len(images)} resim",
    )

    result = []
    for index, image in enumerate(images, 1):
        label = paths["labels"] / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Label bulunamadi: {label}")

        # YAML olmadigi icin class ID'leri sadece sayisal olarak dogrula.
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
                f"{root.name}/{split} tarama",
                index,
                len(images),
                f"{index}/{len(images)} resim",
                finish=index == len(images),
            )
    return result


def manifest_selected_split(root, split):
    """Yalnizca secilen train/val/test splitini Pair listesine cevirir."""
    if split is None:
        return manifest(root, require_yaml=True)

    dirs = split_dirs(root, create=False)
    paths = dirs[split]
    if not paths["images"].is_dir() or not paths["labels"].is_dir():
        raise FileNotFoundError(
            f"{root} icinde {split}/images ve {split}/labels bulunamadi."
        )

    _, names, _ = load_yaml(root, required=True)
    known = set(names)
    images = image_files(paths["images"])

    show_progress(
        f"{root.name}/{split} tarama",
        0,
        len(images),
        f"0/{len(images)} resim",
    )
    result = []
    for index, image in enumerate(images, 1):
        label = paths["labels"] / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Label bulunamadi: {label}")
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
                f"{root.name}/{split} tarama",
                index,
                len(images),
                f"{index}/{len(images)} resim",
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
        "Ana datasete kopyalanacak KAYNAK klasoru/spliti secin:"
    )
    if not sources:
        raise ValueError("En az bir kaynak dataset secilmelidir.")

    source_roots = {root for root, _ in sources}
    source_info = []
    for source, selected_split in sources:
        # Yalnizca train/val/valid/test secildiyse, data.yaml olmadan
        # devam edilebilmesine izin ver. Class ID esleme yapilamayacagi icin
        # bu durumda kaynak class isimleri bos kabul edilir ve hedef secimine
        # gecilir.
        if selected_split is not None:
            yaml_path = source / DATA_YAML_NAME
            if not yaml_path.is_file():
                if not ask_confirm(
                    f"{source} icinde data.yaml bulunamadi. "
                    "data.yaml olmadan devam etmek ister misiniz?",
                    False,
                ):
                    raise RuntimeError(
                        f"data.yaml bulunmadigi icin islem iptal edildi: {source}"
                    )

                source_names = {}
                pairs = manifest_selected_split_without_yaml(source, selected_split)
                source_info.append((source, selected_split, source_names, pairs))
                continue

        # Dataset kokunun tamami secildiyse data.yaml zorunludur.
        if selected_split is None:
            repair_missing_pairs(source)
        # selected_split != None ise kaynak klasorune kesinlikle yeni
        # train/val/test klasoru olusturulmaz.

        try:
            _, source_names, _ = load_yaml(source)
        except ValueError as exc:
            # data.yaml mevcut ama names/class listesi bos veya gecersizse,
            # yalnizca tek bir split secildiginde kullaniciya devam etme sansi ver.
            if selected_split is None:
                raise

            if not ask_confirm(
                f"{source / DATA_YAML_NAME} icinde class isimleri bos veya gecersiz. "
                "data.yaml class isimleri olmadan devam etmek ister misiniz?",
                False,
            ):
                raise RuntimeError(
                    f"data.yaml class isimleri gecersiz oldugu icin islem iptal edildi: {source}"
                )

            source_names = {}
            pairs = manifest_selected_split_without_yaml(source, selected_split)
            source_info.append((source, selected_split, source_names, pairs))
            continue

        pairs = manifest_selected_split(source, selected_split)
        source_info.append((source, selected_split, source_names, pairs))

    create_new = ask_confirm("Yeni bir hedef/ana dataset klasoru olusturulsun mu?", False)
    destination_is_new = False
    rebuild_destination_yaml = False
    if create_new:
        destination_parent = choose_directory(
            "Yeni hedef klasorun olusturulacagi ust dizini secin:", BASE_DIRECTORY
        )
        destination_name = validate_new_dataset_name(ask_text("Yeni hedef klasorun adi:"))
        destination = destination_parent / destination_name
        if destination.exists():
            raise FileExistsError(f"Bu isimde dosya/klasor zaten var: {destination}")
        destination_data, destination_names = {}, {}
        destination_yaml = destination / DATA_YAML_NAME
        destination_is_new = True
        rebuild_destination_yaml = True
    else:
        destination = choose_destination_folder(
            "Mevcut hedef/ana dataset:",
            source_roots,
        )

        # ONCE: kullanicinin secimini images+labels mantigiyla coz.
        # Bu blokta train/val/test OLUSTURULMAZ.
        direct_target = resolve_images_labels_target(destination)

        if direct_target is not None:
            flat_destination_mode = "direct"
            destination_images = direct_target["images"]
            destination_labels = direct_target["labels"]

            if direct_target["mode"] == "direct_missing_labels":
                if not ask_confirm(
                    f"{destination_images} ile ayni dizinde labels klasoru yok. "
                    "labels klasorunu olusturup label dosyalarini buraya kopyalamak ister misiniz?",
                    False,
                ):
                    raise RuntimeError(
                        "labels klasoru olmadigi icin islem durduruldu."
                    )
                destination_labels.mkdir(parents=True, exist_ok=True)

            print(
                f"Hedef images : {destination_images}\n"
                f"Hedef labels : {destination_labels}\n"
                "Bu hedef icine train/val/test klasorleri OLUSTURULMAYACAK."
            )

            # Flat/direct hedeflerde split_dirs ve repair_missing_pairs kesinlikle
            # kullanilmaz. Yalnizca secilen images/labels cifti kullanilir.
            destination_yaml = direct_target["root"] / DATA_YAML_NAME
            yaml_exists = destination_yaml.is_file()

            destination_data, destination_names, _ = load_yaml(
                direct_target["root"], required=False
            )

            if yaml_exists:
                rebuild_destination_yaml = ask_confirm(
                    "Hedefte data.yaml zaten var. Secilen kaynaklarin data.yaml class "
                    "siralarina gore yeniden olusturulsun mu?",
                    False,
                )
            else:
                # Hedefte mevcut image/label ciftleri varsa manifest() kullanma;
                # o fonksiyon split yapisini varsayabilir. Dogrudan klasorleri tara.
                existing_images = image_files(destination_images)
                existing_labels = list(destination_labels.glob("*.txt"))
                if existing_images or existing_labels:
                    if not destination_names:
                        print(
                            "Hedefte data.yaml yok ve mevcut images/labels dosyalari var. "
                            "Mevcut ID anlamlari korunacak; yeni data.yaml ancak class "
                            "isimleri belirlenebiliyorsa yazilacak."
                        )
                rebuild_destination_yaml = True

        else:
            # Buraya ancak kullanici gercek bir dataset kokunu / normal hedefi
            # sectiginde gelinir. Flat/split hedefe dokunulmaz.
            flat_destination_mode = None

            destination_data, destination_names, destination_yaml = load_yaml(
                destination, required=False
            )

            # Normal hedefte eksik split yapisini burada olusturabiliriz.
            # Bu, tek images/labels secimi icin ASLA calismaz.
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
                    "Hedefte data.yaml zaten var. Secilen kaynaklarin data.yaml class "
                    "siralarina gore yeniden olusturulsun mu? Mevcut label ID'leri de "
                    "guvenli sekilde degistirilir.",
                    False,
                )
            else:
                existing_pairs = manifest(destination, require_yaml=False)
                if existing_pairs:
                    raise RuntimeError(
                        "Hedefte data.yaml yok fakat mevcut resim/label var. "
                        "Eski ID anlamlari bilinemedigi icin otomatik data.yaml "
                        "olusturmak guvenli degil."
                    )
                rebuild_destination_yaml = True
                print(
                    "Hedefte data.yaml yok; secilen kaynak data.yaml dosyalarindan "
                    "olusturulacak."
                )


    mode_choices = [questionary.Choice(
        "data.yaml adlarina gore eslestir; yenileri sona ekle (onerilen)", "names")]
    if not destination_is_new and not rebuild_destination_yaml:
        mode_choices.extend([
            questionary.Choice(
                "Her kaynaktaki tum dolu kutulari secilecek tek hedef ID'ye yap", "single"),
            questionary.Choice(
                "ID'leri degistirmeden kopyala (yalniz birebir ayni anlamdaysa)", "raw"),
        ])
    mode = ask_select(
        "Class ID esleme yontemi:",
        mode_choices,
    )

    output_names = {} if rebuild_destination_yaml else dict(destination_names)
    mappings = {}
    for source, selected_split, source_names, source_pairs in source_info:
        source_key = (source, selected_split)
        if mode == "names":
            if not source_names:
                # data.yaml olmayan tek-split kaynakta class isimleri bilinmiyor.
                # Bu durumda label ID'lerini oldugu gibi kopyalamak guvenli olan
                # tek davranistir; hedef YAML'da ilgili ID'ler zaten varsa kullanilir.
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
                            f"{source.name}/{selected_split}: data.yaml yok ve "
                            f"hedef data.yaml bu class ID'lerini tanimlamiyor: "
                            f"{missing_ids}. Class isimleri bilinmeden guvenli "
                            "esleme yapilamaz."
                        )
                id_map = {}
                for pair in source_pairs:
                    for cid in pair.class_ids:
                        id_map[cid] = cid
            else:
                id_map, output_names = build_name_map(source_names, output_names)
        elif mode == "single":
            print(f"\nHedef class'lar ({source.name}/{selected_split or 'tum splitler'} icin):")
            for cid in sorted(output_names):
                print(f"  {cid}: {output_names[cid]}")
            new_id = int(ask_text(
                f"{source.name}/{selected_split or 'tum splitler'} icindeki "
                "tum kutularin hedef class ID'si:"
            ))
            if new_id not in output_names:
                raise ValueError(f"ID hedef data.yaml icinde bulunmuyor: {new_id}")
            id_map = {old: new_id for old in source_names}
        else:
            if set(source_names) - set(output_names):
                raise ValueError(f"{source.name}: hedefte bulunmayan ID var; RAW guvensiz.")
            bad = [
                i for i in source_names
                if normalized_name(source_names[i]) != normalized_name(output_names[i])
            ]
            if bad:
                raise ValueError(f"{source.name}: ayni ID farkli class anlaminda: {bad}")
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
                "Hedef data.yaml yeniden olusturulamaz. Mevcut label'larda kullanilan su "
                "class'lar secilen kaynak data.yaml dosyalarinda yok: "
                + ", ".join(missing_used_classes)
            )

    print("\nClass eslemeleri:")
    total_pairs = 0
    for source, selected_split, source_names, pairs in source_info:
        split_text = selected_split or "train/val/test"
        print(f"  [{source.name} -> {split_text}] resim-label cifti={len(pairs)}")
        total_pairs += len(pairs)
        for old in sorted(mappings[(source, selected_split)]):
            new = mappings[(source, selected_split)][old]
            print(f"    {old}:{source_names[old]} -> {new}:{output_names[new]}")

    print("\nOlusacak ana data.yaml class sirasi:")
    for cid in sorted(output_names):
        print(f"  {cid}: {output_names[cid]}")
    print(f"\nHedef: {destination}\nToplam kopyalanacak cift: {total_pairs}")
    if not ask_confirm("Birlestirmeyi baslat?", False):
        return

    if destination_is_new:
        split_dirs(destination, create=True)
    else:
        backup = backup_metadata(destination, "merge")
        if backup:
            print("Hedef label/YAML yedegi:", backup)

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
            f"Yeni data.yaml sirasi icin degistirilen mevcut hedef label: "
            f"{remapped_existing}"
        )

    # Direct images/labels hedefinde split dizinleri olusturulmaz.
    if not destination_is_new and flat_destination_mode == "direct":
        dirs = None
    elif destination_is_new:
        dirs = split_dirs(destination, create=True)
    else:
        dirs = split_dirs(destination, create=True)

    renamed = negatives = 0
    copied = 0
    copy_progress_step = max(1, total_pairs // 100)
    show_progress("Kopyalama", 0, total_pairs, f"0/{total_pairs} cift")

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
                    "Kopyalama",
                    copied,
                    total_pairs,
                    f"{copied}/{total_pairs} cift",
                    finish=copied == total_pairs,
                )

    print("data.yaml yaziliyor ve birlestirilen dataset dogrulaniyor...")
    if output_names:
        write_yaml(destination_yaml, destination_data, output_names)
    else:
        # Class isimleri bilinmiyorsa bos/hatali data.yaml olusturmak yerine
        # YAML'siz devam et. Kullanici daha once bu duruma izin vermistir.
        print("Class isimleri bilinmiyor; bos data.yaml olusturulmayacak.")

    if not destination_is_new and flat_destination_mode == "direct":
        # Dogrudan secilen images/labels hedefini validate et.
        validate_flat_dataset(
            direct_target["root"],
            False,
            bool(output_names),
        )
    else:
        validate_dataset(destination, False, bool(output_names))
    print(
        f"Bitti: kopyalanan={copied}, yeniden adlandirilan={renamed}, "
        f"negatif={negatives}"
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
    ratios = tuple(float(ask_text(f"{name.capitalize()} yuzdesi:", str(int(default))))
                   for name, default in zip(SPLITS, DEFAULT_RATIOS))
    if any(x < 0 or x > 100 for x in ratios) or not math.isclose(sum(ratios), 100, abs_tol=1e-7):
        raise ValueError("Oranlar 0-100 arasinda ve toplami tam %100 olmali.")
    return ratios


def _impl_redistribute_datasets():
    declared = int(ask_text("Kac class/dataset klasorunuz var?", "1"))
    if declared < 1:
        raise ValueError("Klasor sayisi en az 1 olmali.")
    single = choose_many_datasets(
        "Tek class iceren dataset klasorlerini secin:", allow_empty=True
    )
    multi = choose_many_datasets(
        "Cok class iceren dataset klasorlerini secin:", exclude=set(single), allow_empty=True
    )
    selected = single + multi
    if len(selected) != declared:
        raise ValueError(f"{declared} bildirdiniz, {len(selected)} klasor sectiniz.")
    ratios = ask_ratios()
    plans = []
    for root in selected:
        repair_missing_pairs(root)
        pairs = manifest(root, root in multi)
        split_dirs(root, create=True)
        assigned, targets, class_targets = multilabel_assignment(pairs, ratios, root in multi, SEED)
        print(f"\n{root.name}: toplam={len(pairs)}, hedef={targets}")
        if root in multi:
            _, names, _ = load_yaml(root)
            for cid in sorted(class_targets):
                print(f"  {cid}:{names[cid]} resim hedefi -> {class_targets[cid]}")
        moves = sum(p.source_split != s for s, items in assigned.items() for p in items)
        print("  Tasinacak cift:", moves)
        plans.append((root, assigned, root in multi))
    if not ask_confirm("Dagitim planlarini uygula?", False):
        return
    for root, assigned, is_multi in plans:
        backup = backup_metadata(root, "split")
        if backup:
            print(f"{root.name} label/YAML yedegi: {backup}")
        temp = root / f"_split_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        temp.mkdir()
        staged = []
        total = sum(len(items) for items in assigned.values())
        show_progress("Yeniden bolusturme - gecici tasima", 0, total, f"0/{total} cift")
        for split, items in assigned.items():
            for pair in items:
                index = len(staged)
                ti, tl = temp / f"{index:09d}{pair.image.suffix}", temp / f"{index:09d}.txt"
                shutil.move(str(pair.image), ti); shutil.move(str(pair.label), tl)
                staged.append((split, pair.image.stem, pair.image.suffix, ti, tl))
                done = len(staged)
                if done == total or done % max(1, total // 100) == 0:
                    show_progress(
                        "Yeniden bolusturme - gecici tasima", done, total,
                        f"{done}/{total} cift", finish=done == total,
                    )
        dirs = split_dirs(root, create=True)
        show_progress("Yeniden bolusturme - yerlestirme", 0, total, f"0/{total} cift")
        for index, (split, old_stem, suffix, ti, tl) in enumerate(staged, 1):
            target = dirs[split]
            stem, _ = unique_stem(old_stem, root.name, target["images"], target["labels"])
            shutil.move(str(ti), target["images"] / f"{stem}{suffix}")
            shutil.move(str(tl), target["labels"] / f"{stem}.txt")
            if index == total or index % max(1, total // 100) == 0:
                show_progress(
                    "Yeniden bolusturme - yerlestirme", index, total,
                    f"{index}/{total} cift", finish=index == total,
                )
        temp.rmdir()
        validate_dataset(root, False, is_multi)
    print("Oranli yeniden dagitim tamamlandi.")


def redistribute_datasets():
    try:
        return _impl_redistribute_datasets()
    except BackMenu:
        return None


def conversion_candidates():
    """Eski split/images duzenindeki klasorleri donusum icin listeler."""
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
    """Donusumden once butun resim-label eslerini ve etiketleri kontrol eder."""
    if (root / "val").exists() and (root / "valid").exists():
        raise RuntimeError("Hem val hem valid klasoru var. Birini secip tek isimde birlestirin.")
    val_source = "val" if (root / "val").is_dir() else "valid"
    aliases = {"train": "train", "val": val_source, "test": "test"}
    _, names, _ = load_yaml(root, required=True)
    known = set(names)
    summary = {}
    for final_split, source_split in aliases.items():
        images_dir = root / source_split / "images"
        labels_dir = root / source_split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Eksik klasor: {images_dir} veya {labels_dir}")
        images = image_files(images_dir)
        labels = sorted(labels_dir.glob("*.txt"))
        image_stems = [p.stem for p in images]
        if len(image_stems) != len(set(image_stems)):
            raise RuntimeError(f"Ayni kok ada sahip birden fazla resim var: {images_dir}")
        label_stems = {p.stem for p in labels}
        missing_labels = set(image_stems) - label_stems
        missing_images = label_stems - set(image_stems)
        if missing_labels or missing_images:
            raise RuntimeError(
                f"{source_split} esleme hatasi; label'i olmayan resim={sorted(missing_labels)[:10]}, "
                f"resmi olmayan label={sorted(missing_images)[:10]}"
            )
        boxes = negatives = 0
        for label in labels:
            lines, _ = parse_label(label, known)
            boxes += len(lines)
            negatives += int(not lines)
        summary[final_split] = (len(images), len(labels), boxes, negatives)
    return aliases, summary


def validate_final_layout(root):
    """images/split + labels/split bicimindeki ZIP'e hazir yapinin kontrolu."""
    _, names, _ = load_yaml(root, required=True)
    known = set(names)
    print(f"\nSon yapi dogrulamasi: {root}")
    for split in ("train", "val", "test"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = image_files(images_dir)
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        if image_stems != label_stems:
            raise RuntimeError(
                f"{split}: resim-label eslesmiyor; labelsiz={sorted(image_stems-label_stems)[:10]}, "
                f"resimsiz={sorted(label_stems-image_stems)[:10]}"
            )
        boxes = negatives = 0
        for label in labels:
            lines, _ = parse_label(label, known)
            boxes += len(lines)
            negatives += int(not lines)
        print(
            f"  {split:5s}: resim={len(images)}, label={len(labels)}, "
            f"kutu={boxes}, negatif={negatives} [OK]"
        )


def _impl_convert_to_zip_layout():
    root = choose_directory(
        "ZIP'e hazir images/split + labels/split duzenine cevrilecek ANA dataset:",
        BASE_DIRECTORY,
    ).resolve()
    if root not in conversion_candidates_for_root(root):
        raise ValueError(
            "Secilen klasor train/{images,labels}, val veya valid/{images,labels}, "
            "test/{images,labels} calisma duzeninde degil."
        )
    if (root / "images").exists() or (root / "labels").exists():
        raise RuntimeError(
            "Hedefte zaten images veya labels klasoru var. Uzerine yazma riskinden islem durduruldu."
        )
    aliases, summary = inspect_old_layout(root)
    print("\nDonusum plani:")
    for split in ("train", "val", "test"):
        ni, nl, nb, ne = summary[split]
        print(f"  {aliases[split]} -> {split}: resim={ni}, label={nl}, kutu={nb}, negatif={ne}")
    print("\nEski: train/images + train/labels")
    print("Yeni: images/train + labels/train")
    print("UYARI: Bu son paketleme adimidir; sonrasinda 3-5 numarali islemleri calistirmayin.")
    if not ask_confirm("Ana dataset yapisi simdi donusturulsun mu?", False):
        print("Degisiklik yapilmadi.")
        return

    data, names, yaml_path = load_yaml(root, required=True)
    should_save, yaml_backup_name = ask_save_backup(
        "layout donusumu",
        "Mevcut data.yaml dosyasinin",
        f"{root.name}_layout_yaml_backup",
    )
    if should_save:
        yaml_backup = root.parent / yaml_backup_name
        if yaml_backup.exists() and not ask_confirm(f"{yaml_backup} zaten var. Uzerine yazilsin mi?", False):
            print("data.yaml save islemi iptal edildi.")
            return
        with ZipFile(yaml_backup, "w", ZIP_DEFLATED) as archive:
            archive.write(yaml_path, yaml_path.relative_to(root))
        print("data.yaml save dosyasi:", yaml_backup)
    else:
        yaml_backup = None
        print("data.yaml save dosyasi olusturulmadi.")

    # Generate the unique temporary-directory suffix before using it.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp = root / f"_layout_conversion_temp_{stamp}"
    if temp.exists():
        raise FileExistsError(f"Gecici klasor zaten var: {temp}")
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
        print(f"Donusum yarida kaldi. Kurtarma dosyalari burada: {temp}")
        raise

    output = dict(data)
    output["train"] = "images/train"
    output["val"] = "images/val"
    output["test"] = "images/test"
    output.pop("path", None)
    write_yaml(yaml_path, output, names)
    # write_yaml calisma duzenini yazar; final paket yollarini son kez kesinlestir.
    output = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    output["train"], output["val"], output["test"] = (
        "images/train", "images/val", "images/test"
    )
    yaml_path.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")
    validate_final_layout(root)
    print("Ana dataset ZIP'e hazir dizilime donusturuldu.")


def convert_to_zip_layout():
    try:
        return _impl_convert_to_zip_layout()
    except BackMenu:
        return None


def final_layout_candidates():
    """images/{train,val,test} + labels/{train,val,test} yapisindakiler."""
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
    root = choose_directory("ZIP'lenecek ana dataset:", BASE_DIRECTORY).resolve()
    if not (
        (root / DATA_YAML_NAME).is_file()
        and all(
            (root / kind / split).is_dir()
            for kind in ("images", "labels") for split in SPLITS
        )
    ):
        raise ValueError(
            "Secilen klasor ZIP'e uygun images/{train,val,test} + "
            "labels/{train,val,test} yapisinda degil."
        )
    validate_final_layout(root)
    zip_directory = choose_directory("ZIP dosyasinin kaydedilecegi dizin:", root.parent).resolve()
    output = zip_directory / f"{root.name}.zip"
    temporary = zip_directory / f".{root.name}.zip.part"
    if output.exists() and not ask_confirm(
        f"{output.name} zaten var. Dogrulanmis yeni ZIP ile degistirilsin mi?", False
    ):
        print("ZIP olusturma iptal edildi.")
        return
    if temporary.exists():
        temporary.unlink()

    files = [root / DATA_YAML_NAME]
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            files.extend(sorted(p for p in (root / kind / split).rglob("*") if p.is_file()))
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"ZIP dosyasi: {output}")
    print(f"Dosya sayisi: {len(files):,}; ham boyut: {total_bytes / (1024**3):.2f} GB")
    if not ask_confirm("ZIP olusturma baslatilsin mi?", True):
        print("ZIP olusturma iptal edildi.")
        return

    try:
        # ZIP64 buyuk datasetleri destekler. Dusuk sikistirma seviyesi JPG/PNG'de
        # gereksiz CPU harcamadan txt/yaml dosyalarini yine sikistirir.
        with ZipFile(
            temporary, "w", ZIP_DEFLATED, allowZip64=True, compresslevel=1
        ) as archive:
            # Bos split klasorleri de arsivde gorunsun.
            for kind in ("images", "labels"):
                for split in ("train", "val", "test"):
                    archive.writestr(f"{kind}/{split}/", "")
            processed_bytes = 0
            last_percentage = -1
            show_progress(
                "ZIP olusturma", 0, max(1, total_bytes),
                f"0/{len(files)} dosya | 0.00/{total_bytes / (1024**3):.2f} GB",
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
                        "ZIP olusturma", progress_current, max(1, progress_total),
                        f"{index}/{len(files)} dosya | "
                        f"{processed_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GB",
                        finish=index == len(files),
                    )
        # Once gecici ZIP tamamen kapanir; sonra tek adimda asil ada gecirilir.
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            print(f"Yarim ZIP asil dosyanin yerine gecmedi: {temporary}")
        raise

    print("ZIP butunluk kontrolu yapiliyor...")
    with ZipFile(output, "r") as archive:
        bad = archive.testzip()
        required = {"data.yaml", "images/train/", "images/val/", "images/test/",
                    "labels/train/", "labels/val/", "labels/test/"}
        missing = required - set(archive.namelist())
    if bad or missing:
        raise RuntimeError(f"ZIP dogrulamasi basarisiz; bozuk={bad}, eksik={sorted(missing)}")
    print(f"ZIP olusturuldu ve dogrulandi: {output}")


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
            f"Flat dataset icin images ve labels gerekli: {root}"
        )

    _, names, _ = load_yaml(root, required=require_yaml)
    known = set(names) if names else None

    images = image_files(images_dir)
    labels = sorted(labels_dir.glob("*.txt"))
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    errors = []
    errors += [f"{root.name}: {s} resmine ait label yok" for s in image_stems - label_stems]
    errors += [f"{root.name}: {s}.txt label'ina ait resim yok" for s in label_stems - image_stems]

    total_boxes = total_empty = 0
    class_boxes = Counter()

    show_progress("Flat dataset dogrulama", 0, len(labels), f"0/{len(labels)} label")
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
                "Flat dataset dogrulama",
                index,
                len(labels),
                f"{index}/{len(labels)} label",
                finish=index == len(labels),
            )

    print(f"\nDogrulama: {root}")
    print(
        f"  images: {len(images)}, labels: {len(labels)}, "
        f"kutu={total_boxes}, negatif={total_empty}"
    )
    if names:
        for cid in sorted(names):
            print(f"  class {cid}:{names[cid]} kutu={class_boxes[cid]}")

    if errors or len(images) != len(labels):
        raise RuntimeError(
            f"Flat dataset dogrulama basarisiz ({len(errors)} hata):\n"
            + "\n".join(errors[:30])
        )

    print("  SONUC: resim-label esleri ve YOLO etiketleri gecerli.")
    return {"flat": (len(images), len(labels), total_boxes, total_empty)}


def validate_dataset(root, interactive=True, require_yaml=False):
    root = Path(root).resolve()

    if interactive:
        # Do not create directories while validating. Repairing missing
        # image/label pairs is a separate operation and must never turn an
        # invalid layout into an apparently valid empty dataset.
        pass

    layout_paths = validation_split_dirs(root)
    errors = []
    warnings = []

    existing_splits = [
        split for split, paths in layout_paths.items()
        if paths["images"].is_dir() or paths["labels"].is_dir()
    ]

    if not existing_splits:
        raise ValueError(
            "Dataset yapisi taninamadi. Beklenen iki yapi:\n"
            "  1) train/images, train/labels, val/images, val/labels, "
            "test/images, test/labels\n"
            "  2) images/train, images/val, images/test, "
            "labels/train, labels/val, labels/test"
        )

    # Load data.yaml without modifying the dataset.
    _, names, _ = load_yaml(root, required=require_yaml)
    known = set(names) if names else None

    total_images = total_labels = total_boxes = total_empty = 0
    class_boxes = Counter()
    summaries = {}

    all_labels = []
    for split in SPLITS:
        paths = layout_paths[split]
        if paths["labels"].is_dir():
            all_labels.extend(
                sorted(paths["labels"].glob("*.txt"))
            )

    processed_labels = 0
    show_progress(
        "Dataset dogrulama",
        0,
        len(all_labels),
        f"0/{len(all_labels)} label",
    )

    for split in SPLITS:
        paths = layout_paths[split]
        images_dir = paths["images"]
        labels_dir = paths["labels"]

        images = (
            image_files(images_dir)
            if images_dir.is_dir()
            else []
        )
        labels = (
            sorted(labels_dir.glob("*.txt"))
            if labels_dir.is_dir()
            else []
        )

        i_stems = {p.stem for p in images}
        l_stems = {p.stem for p in labels}

        if images_dir.is_dir() and not labels_dir.is_dir():
            errors.append(
                f"{root.name}/{split}: images klasoru var fakat labels klasoru yok"
            )
        if labels_dir.is_dir() and not images_dir.is_dir():
            errors.append(
                f"{root.name}/{split}: labels klasoru var fakat images klasoru yok"
            )

        errors += [
            f"{root.name}/{split}: {s} resmine ait label yok"
            for s in sorted(i_stems - l_stems)
        ]
        errors += [
            f"{root.name}/{split}: {s}.txt label'ina ait resim yok"
            for s in sorted(l_stems - i_stems)
        ]

        boxes = 0
        empty = 0

        for label in labels:
            try:
                lines, counts = parse_label(label, known)
                box_count = len(lines)
                boxes += box_count
                empty += int(not lines)
                class_boxes.update(counts)

            except ValueError as exc:
                errors.append(str(exc))

            processed_labels += 1
            if (
                processed_labels == len(all_labels)
                or processed_labels % max(1, len(all_labels) // 100) == 0
            ):
                show_progress(
                    "Dataset dogrulama",
                    processed_labels,
                    len(all_labels),
                    f"{processed_labels}/{len(all_labels)} label",
                    finish=processed_labels == len(all_labels),
                )

        summaries[split] = (
            len(images),
            len(labels),
            boxes,
            empty,
            paths["layout"],
        )
        total_images += len(images)
        total_labels += len(labels)
        total_boxes += boxes
        total_empty += empty

    print(f"\nDogrulama: {root}")
    print("  Dataset layout kontrolu:")

    layouts = {
        paths["layout"]
        for paths in layout_paths.values()
        if paths["layout"] != "missing"
    }

    if len(layouts) == 1:
        detected_layout = next(iter(layouts))
        print(f"    Bulunan yapi: {detected_layout}")
    elif layouts:
        print(
            "    UYARI: Farkli split'lerde farkli dizin yapilari bulundu: "
            + ", ".join(sorted(layouts))
        )
        warnings.append("Split'ler arasinda farkli dataset dizin yapilari var.")

    for split, summary in summaries.items():
        ni, nl, nb, ne, layout = summary

        if layout == "missing":
            print(
                f"  {split:5s}: resim={ni}, label={nl}, kutu={nb}, "
                f"negatif={ne} [YOK]"
            )
            continue

        status = "OK" if ni == nl and not any(
            f"{root.name}/{split}:" in error for error in errors
        ) else "HATA"

        print(
            f"  {split:5s}: resim={ni}, label={nl}, kutu={nb}, "
            f"negatif={ne} [{status}]"
        )

    if names:
        for cid in sorted(names):
            print(
                f"  class {cid}:{names[cid]} "
                f"kutu={class_boxes[cid]}"
            )

    print(
        f"  TOPLAM: resim={total_images}, label={total_labels}, "
        f"kutu={total_boxes}, negatif={total_empty}"
    )

    # Final label-format check: every YOLO detection row must contain
    # exactly five whitespace-separated values:
    # class_id x_center y_center width height
    invalid_column_rows = []
    for split in SPLITS:
        labels_dir = layout_paths[split]["labels"]
        if not labels_dir.is_dir():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            try:
                with label_path.open("r", encoding="utf-8") as handle:
                    for line_number, raw_line in enumerate(handle, 1):
                        line = raw_line.strip()
                        if not line:
                            continue
                        column_count = len(line.split())
                        if column_count != 5:
                            invalid_column_rows.append(
                                (split, label_path.name, line_number, column_count)
                            )
            except OSError as exc:
                errors.append(
                    f"{label_path}: label dosyasi okunamadi: {exc}"
                )

    if invalid_column_rows:
        print(
            "\n  UYARI: 5 sutunlu YOLO label formatina uymayan satirlar:"
        )
        for split, label_name, line_number, column_count in invalid_column_rows:
            print(
                f"    - {split}/{label_name}:{line_number} "
                f"-> {column_count} deger (beklenen: 5)"
            )
    else:
        print(
            "\n  LABEL FORMAT KONTROLU: Tum label satirlari 5 deger iceriyor."
        )

    # Explicitly explain the two supported layouts and the conversion path.
    if "images/split + labels/split" in layouts:
        print(
            "\n  BILGI: Datasetiniz egitim icin kullanilan "
            "images/split + labels/split duzenine sahip."
        )
    elif "split/images + split/labels" in layouts:
        print(
            "\n  UYARI: Datasetiniz split/images + split/labels "
            "duzeninde."
        )
        print(
            "  Dataseti egitim icin uygun images/split + labels/split "
            "duzenine cevirmek icin:"
        )
        print(
            "  Ana menu -> Ana dataset yapisini images/split + "
            "labels/split duzenine cevir"
        )

    if warnings:
        print("\n  UYARILAR:")
        for warning in warnings:
            print(f"    - {warning}")

    if invalid_column_rows:
        errors.extend(
            [
                f"{split}/{label_name}:{line_number}: "
                f"{column_count} deger (beklenen: 5)"
                for split, label_name, line_number, column_count
                in invalid_column_rows
            ]
        )

    if errors:
        print(
            f"\n  SONUC: {len(errors)} hata bulundu. "
            "Dataset egitim icin guvenli kabul edilmemeli."
        )
        print("  Ilk hatalar:")
        for error in errors[:30]:
            print(f"    - {error}")
        return summaries

    if total_images != total_labels:
        print(
            "\n  SONUC: Resim ve label sayilari esit degil. "
            "Dataset egitim icin uygun degil."
        )
        return summaries

    print(
        "\n  SONUC: Resim-label eslesmeleri ve YOLO etiketleri "
        "gecerli gorunuyor."
    )
    return summaries


def _impl_validation_menu():
    roots = choose_many_datasets("Dogrulanacak datasetleri secin:")
    if not roots:
        raise ValueError("En az bir dataset secilmeli.")
    for root in roots:
        validate_dataset(root, True, False)


def validation_menu():
    try:
        return _impl_validation_menu()
    except BackMenu:
        return None


def _impl_create_empty_labels_for_images():
    images_dir = choose_directory(
        "Bos label olusturulacak fotograflarin bulundugu images klasoru:",
        BASE_DIRECTORY,
    ).resolve()
    images = image_files(images_dir)
    if not images:
        raise FileNotFoundError(f"Secilen dizinde desteklenen resim bulunamadi: {images_dir}")
    stems = [image.stem for image in images]
    duplicate_stems = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicate_stems:
        raise RuntimeError(
            "Ayni kok ada sahip birden fazla resim var; tek label adi iki resme ait olamaz: "
            + ", ".join(duplicate_stems[:20])
        )

    labels_dir = images_dir.parent / "labels"
    existing_items = list(labels_dir.iterdir()) if labels_dir.is_dir() else []
    existing_labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    matching_existing = [labels_dir / f"{image.stem}.txt" for image in images]
    overwrite_count = sum(label.is_file() for label in matching_existing)

    print("\nBos label olusturma plani:")
    print("  Images:", images_dir)
    print("  Labels:", labels_dir)
    print(f"  Resim={len(images)}, mevcut eslesen label={overwrite_count}")
    if existing_items:
        if not ask_confirm(
            "Labels klasorunun ici dolu. Her resimle ayni ada sahip label dosyalari "
            "bos icerikle yazilacak. Uzerine yazmami ister misiniz?",
            False,
        ):
            print("Bos label olusturma iptal edildi.")
            return
    elif not ask_confirm(
        "Her resim icin ayni ada sahip bos .txt label dosyasi olusturulsun mu?",
        False,
    ):
        print("Bos label olusturma iptal edildi.")
        return

    if existing_labels:
        should_save, backup_name = ask_save_backup(
            "bos label olusturma",
            "Mevcut label dosyalarinin",
            "labels_empty_backup",
        )
        if should_save:
            backup = images_dir.parent / backup_name
            if backup.exists() and not ask_confirm(f"{backup} zaten var. Uzerine yazilsin mi?", False):
                print("Label save islemi iptal edildi.")
                return
            with ZipFile(backup, "w", ZIP_DEFLATED) as archive:
                show_progress("Mevcut label save", 0, len(existing_labels), f"0/{len(existing_labels)}")
                for index, label in enumerate(existing_labels, 1):
                    archive.write(label, f"labels/{label.name}")
                    if index == len(existing_labels) or index % max(1, len(existing_labels) // 100) == 0:
                        show_progress(
                            "Mevcut label save", index, len(existing_labels),
                            f"{index}/{len(existing_labels)}", finish=index == len(existing_labels),
                        )
            print("Mevcut label save dosyasi:", backup)
        else:
            print("Mevcut label save dosyasi olusturulmadi.")

    labels_dir.mkdir(parents=True, exist_ok=True)
    show_progress("Bos label olusturma", 0, len(images), f"0/{len(images)} label")
    for index, image in enumerate(images, 1):
        (labels_dir / f"{image.stem}.txt").write_text("", encoding="utf-8")
        if index == len(images) or index % max(1, len(images) // 100) == 0:
            show_progress(
                "Bos label olusturma", index, len(images), f"{index}/{len(images)} label",
                finish=index == len(images),
            )
    invalid = [label for label in matching_existing if not label.is_file() or label.stat().st_size != 0]
    if invalid:
        raise RuntimeError("Bos label dogrulamasi basarisiz: " + ", ".join(map(str, invalid[:20])))
    print(f"Tamamlandi: {len(images)} resim icin bos label hazirlandi: {labels_dir}")


def create_empty_labels_for_images():
    try:
        return _impl_create_empty_labels_for_images()
    except BackMenu:
        return None


def main():
    print(f"\nYOLO Dataset Yonetim Araci\nCalisma dizini: {BASE_DIRECTORY}\n")

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
                            "Linux dagitiminiz:",
                            [questionary.Choice("(1) Arch / Arch tabanli", "arch"),
                             questionary.Choice("(2) Debian / Ubuntu tabanli", "debian"),
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
                        "Yapilacak islem:",
                        [questionary.Choice("(1) Kutu sec / YOLO box ciz", "annotate"),
                         questionary.Choice("(2) Fotograflar icin bos/negatif label olustur", "empty_labels"),
                         questionary.Choice("(3) Class'lari filtrele/azalt", "filter"),
                         questionary.Choice("(3) Datasetleri ana dataset icine birlestir", "merge"),
                         questionary.Choice("(4) Train/val/test oranlariyla yeniden bolustur", "split"),
                         questionary.Choice("(5) Ana dataseti images/split + labels/split duzenine cevir", "convert"),
                         questionary.Choice("(6) Datasetleri yalnizca kontrol et", "validate"),
                         questionary.Choice("(7) Ana dataseti ZIP dosyasi yap", "zip"),
                         questionary.Choice("(0) Cikis", "exit")],
                    )

                    if action == "exit":
                        print("Cikis yapildi.")
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
                            "Linux dagitiminiz:",
                            [questionary.Choice("(1) Arch / Arch tabanli", "arch"),
                             questionary.Choice("(2) Debian / Ubuntu tabanli", "debian"),
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
        print("\nGeri donulecek bir onceki menu bulunmuyor.")
    except KeyboardInterrupt:
        print("\nIslem kullanici tarafindan iptal edildi.")
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nHATA [{type(exc).__name__}]: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
