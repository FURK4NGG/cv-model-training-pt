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
    value = questionary.select(message, choices=choices, use_shortcuts=False).ask()
    if value is None:
        raise KeyboardInterrupt
    return value


def ask_checkbox(message, choices):
    value = questionary.checkbox(
        message, choices=choices,
        instruction="(Yon tuslari: gezin, Space: sec/kaldir, Enter: onayla)",
    ).ask()
    if value is None:
        raise KeyboardInterrupt
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
    SELECTED_OS = ask_select(
        "Hangi isletim sisteminde calistiriyorsunuz?",
        [questionary.Choice("(1) Windows", "windows"),
         questionary.Choice("(2) Linux", "linux")],
    )
    if SELECTED_OS == "linux":
        SELECTED_LINUX_DISTRO = ask_select(
            "Linux dagitiminiz:",
            [questionary.Choice("(1) Arch / Arch tabanli", "arch"),
             questionary.Choice("(2) Debian / Ubuntu tabanli", "debian"),
             questionary.Choice("(3) Fedora", "fedora")],
        )
    else:
        SELECTED_LINUX_DISTRO = None
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
        raise ValueError("Bu klasor bu islemde tekrar veya hedef olarak secilemez.")
    if not looks_like_dataset(folder):
        raise ValueError(
            "Secilen klasor train/val/test altinda images veya labels iceren bir dataset degil: "
            f"{folder}"
        )
    return folder


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
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


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
        show_progress("Yedekleme", 0, len(files), f"0/{len(files)} dosya")
        for index, file in enumerate(files, 1):
            archive.write(file, file.relative_to(root))
            if index == len(files) or index % max(1, len(files) // 100) == 0:
                show_progress(
                    "Yedekleme", index, len(files), f"{index}/{len(files)} dosya",
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
    if names:
        new_id = ask_select(
            "Butun dolu label satirlari hangi class'a donusturulsun?",
            [questionary.Choice(f"{cid}: {names[cid]}", cid) for cid in sorted(names)],
        )
        target_name = names[new_id]
    else:
        print("data.yaml icinde secilebilecek class bulunamadi; hedefi elle girmeniz gerekiyor.")
        raw_id = ask_text("Butun dolu label satirlarinin yeni class ID degeri:")
        try:
            new_id = int(raw_id)
        except ValueError as exc:
            raise ValueError("Class ID, 0 veya daha buyuk bir tam sayi olmalidir.") from exc
        if new_id < 0:
            raise ValueError("Class ID negatif olamaz.")
        target_name = ask_text("Bu hedef ID'nin class adi:")
        if not target_name:
            raise ValueError("Hedef class adi bos olamaz.")
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
        new_text = rewrite_label(lines, {old_id: new_id for old_id in counts})
        old_text = pair.label.read_text(encoding="utf-8-sig")
        if new_text != old_text:
            plans.append((pair.label, new_text))
        else:
            already_correct += int(bool(lines))

    print("\nToplu class ID degistirme plani:")
    for old_id in sorted(old_id_counts):
        old_name = names.get(old_id, "<data.yaml'da tanimli degil>")
        print(f"  {old_id}:{old_name} -> {new_id}:{target_name} (kutu={old_id_counts[old_id]})")
    print(
        f"Toplam kutu={box_count}, degisecek label={len(plans)}, "
        f"zaten dogru dolu label={already_correct}, bos/negatif label={empty_count}"
    )
    if not ask_confirm("Butun dolu label class ID'leri degistirilsin mi?", False):
        print("Degisiklik yapilmadi.")
        return

    backup = backup_metadata(root, "class_id")
    if backup:
        print("Label/YAML yedegi:", backup)
    show_progress("Label ID degistirme", 0, len(plans), f"0/{len(plans)} label")
    for index, (label, text) in enumerate(plans, 1):
        label.write_text(text, encoding="utf-8")
        if index == len(plans) or index % max(1, len(plans) // 100) == 0:
            show_progress(
                "Label ID degistirme", index, len(plans), f"{index}/{len(plans)} label",
                finish=index == len(plans),
            )
    write_yaml(yaml_path, data, {new_id: target_name})
    validate_dataset(root, interactive=False, require_yaml=True)
    print(
        f"Toplu class ID degistirme tamamlandi. Butun dolu kutular: "
        f"{new_id}:{target_name}"
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


def filter_classes():
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


def merge_datasets():
    sources = choose_ordered_datasets(
        "Ana datasete kopyalanacak KAYNAK klasorleri secin:"
    )
    if not sources:
        raise ValueError("En az bir kaynak dataset secilmelidir.")

    source_info = []
    for source in sources:
        repair_missing_pairs(source)
        _, source_names, _ = load_yaml(source)
        source_info.append((source, source_names, manifest(source, True)))

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
        destination = choose_destination_folder("Mevcut hedef/ana dataset:", set(sources))
        required_paths = [
            destination / split / kind
            for split in SPLITS for kind in ("images", "labels")
        ]
        if all(path.is_dir() for path in required_paths):
            if ask_confirm(
                "Hedefte train/val/test images/labels klasorleri zaten var. "
                "Ayni klasor yapisi yeniden olusturulsun mu? Mevcut dosyalar silinmez.",
                False,
            ):
                split_dirs(destination, create=True)
        else:
            split_dirs(destination, create=True)
            print("Hedefte eksik train/val/test images/labels klasorleri olusturuldu.")
        repair_missing_pairs(destination)
        yaml_exists = (destination / DATA_YAML_NAME).is_file()
        destination_data, destination_names, destination_yaml = load_yaml(
            destination, required=False
        )
        if yaml_exists:
            rebuild_destination_yaml = ask_confirm(
                "Hedefte data.yaml zaten var. Secilen kaynaklarin data.yaml class siralarina "
                "gore yeniden olusturulsun mu? Mevcut label ID'leri de guvenli sekilde degistirilir.",
                False,
            )
        else:
            existing_pairs = manifest(destination, require_yaml=False)
            if existing_pairs:
                raise RuntimeError(
                    "Hedefte data.yaml yok fakat mevcut resim/label var. Eski ID anlamlari "
                    "bilinemedigi icin otomatik data.yaml olusturmak guvenli degil."
                )
            rebuild_destination_yaml = True
            print("Hedefte data.yaml yok; secilen kaynak data.yaml dosyalarindan olusturulacak.")

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

    # Yeni veya yeniden olusturulan YAML'da secim sirasi 0'dan baslar.
    # Mevcut YAML korunuyorsa onun class sirasi once gelir, yeni class'lar eklenir.
    output_names = {} if rebuild_destination_yaml else dict(destination_names)
    mappings = {}
    for source, source_names, _ in source_info:
        if mode == "names":
            id_map, output_names = build_name_map(source_names, output_names)
        elif mode == "single":
            print(f"\nHedef class'lar ({source.name} icin):")
            for cid in sorted(output_names):
                print(f"  {cid}: {output_names[cid]}")
            new_id = int(ask_text(f"{source.name} icindeki tum kutularin hedef class ID'si:"))
            if new_id not in output_names:
                raise ValueError(f"ID hedef data.yaml icinde bulunmuyor: {new_id}")
            id_map = {old: new_id for old in source_names}
        else:
            if set(source_names) - set(output_names):
                raise ValueError(f"{source.name}: hedefte bulunmayan ID var; RAW guvensiz.")
            bad = [i for i in source_names if normalized_name(source_names[i]) !=
                   normalized_name(output_names[i])]
            if bad:
                raise ValueError(f"{source.name}: ayni ID farkli class anlaminda: {bad}")
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
                "Hedef data.yaml yeniden olusturulamaz. Mevcut label'larda kullanilan su "
                "class'lar secilen kaynak data.yaml dosyalarinda yok: "
                + ", ".join(missing_used_classes)
            )

    print("\nClass eslemeleri:")
    total_pairs = 0
    for source, source_names, pairs in source_info:
        print(f"  [{source.name}] resim-label cifti={len(pairs)}")
        total_pairs += len(pairs)
        for old in sorted(mappings[source]):
            new = mappings[source][old]
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
        print(f"Yeni data.yaml sirasi icin degistirilen mevcut hedef label: {remapped_existing}")
    dirs = split_dirs(destination, create=True)
    renamed = negatives = 0
    copied = 0
    copy_progress_step = max(1, total_pairs // 100)
    show_progress("Kopyalama", 0, total_pairs, f"0/{total_pairs} cift")
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
                # ID zaten dogruysa label icerigine dokunmadan kopyala.
                shutil.copy2(pair.label, destination_label)
            else:
                destination_label.write_text(text, encoding="utf-8")
            copied += 1
            renamed += changed
            negatives += not text.strip()
            if copied % copy_progress_step == 0 or copied == total_pairs:
                show_progress(
                    "Kopyalama", copied, total_pairs,
                    f"{copied}/{total_pairs} cift",
                    finish=copied == total_pairs,
                )
    print("data.yaml yaziliyor ve birlestirilen dataset dogrulaniyor...")
    write_yaml(destination_yaml, destination_data, output_names)
    validate_dataset(destination, False, True)
    print(f"Bitti: kopyalanan={copied}, yeniden adlandirilan={renamed}, negatif={negatives}")


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


def redistribute_datasets():
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


def convert_to_zip_layout():
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
    print("UYARI: Bu son paketleme adimidir; sonrasinda 1-3 numarali islemleri calistirmayin.")
    if not ask_confirm("Ana dataset yapisi simdi donusturulsun mu?", False):
        print("Degisiklik yapilmadi.")
        return

    data, names, yaml_path = load_yaml(root, required=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    yaml_backup = root.parent / f"{root.name}_layout_yaml_backup_{stamp}.zip"
    with ZipFile(yaml_backup, "w", ZIP_DEFLATED) as archive:
        archive.write(yaml_path, yaml_path.relative_to(root))
    print("data.yaml yedegi:", yaml_backup)

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


def create_dataset_zip():
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
    show_progress("Dataset dogrulama", 0, len(all_labels), f"0/{len(all_labels)} label")
    for split, paths in split_dirs(root, create=True).items():
        images, labels = image_files(paths["images"]), sorted(paths["labels"].glob("*.txt"))
        i_stems, l_stems = {p.stem for p in images}, {p.stem for p in labels}
        errors += [f"{root.name}/{split}: {s} resmine ait label yok" for s in i_stems-l_stems]
        errors += [f"{root.name}/{split}: {s}.txt label'ina ait resim yok" for s in l_stems-i_stems]
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
                    "Dataset dogrulama", processed_labels, len(all_labels),
                    f"{processed_labels}/{len(all_labels)} label",
                    finish=processed_labels == len(all_labels),
                )
        summaries[split] = len(images), len(labels), boxes, empty
        total_images += len(images); total_labels += len(labels)
        total_boxes += boxes; total_empty += empty
    print(f"\nDogrulama: {root}")
    for split, (ni, nl, nb, ne) in summaries.items():
        print(f"  {split:5s}: resim={ni}, label={nl}, kutu={nb}, negatif={ne} "
              f"[{'OK' if ni == nl else 'HATA'}]")
    if names:
        for cid in sorted(names):
            print(f"  class {cid}:{names[cid]} kutu={class_boxes[cid]}")
    print(f"  TOPLAM: resim={total_images}, label={total_labels}, "
          f"kutu={total_boxes}, negatif={total_empty}")
    if errors or total_images != total_labels:
        raise RuntimeError(f"Dogrulama basarisiz ({len(errors)} hata):\n" + "\n".join(errors[:30]))
    print("  SONUC: resim-label esleri ve YOLO etiketleri gecerli.")
    return summaries


def validation_menu():
    roots = choose_many_datasets("Dogrulanacak datasetleri secin:")
    if not roots:
        raise ValueError("En az bir dataset secilmeli.")
    for root in roots:
        validate_dataset(root, True, False)


def create_empty_labels_for_images():
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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = images_dir.parent / f"labels_empty_backup_{stamp}.zip"
        with ZipFile(backup, "w", ZIP_DEFLATED) as archive:
            show_progress("Mevcut label yedegi", 0, len(existing_labels), f"0/{len(existing_labels)}")
            for index, label in enumerate(existing_labels, 1):
                archive.write(label, f"labels/{label.name}")
                if index == len(existing_labels) or index % max(1, len(existing_labels) // 100) == 0:
                    show_progress(
                        "Mevcut label yedegi", index, len(existing_labels),
                        f"{index}/{len(existing_labels)}", finish=index == len(existing_labels),
                    )
        print("Mevcut label yedegi:", backup)

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


def main():
    print(f"\nYOLO Dataset Yonetim Araci\nCalisma dizini: {BASE_DIRECTORY}\n")
    configure_platform()
    ensure_structure_confirmation()
    action = ask_select(
        "Yapilacak islem:",
        [questionary.Choice("(1) Fotograflar icin bos/negatif label olustur", "empty_labels"),
         questionary.Choice("(2) Class'lari filtrele/azalt", "filter"),
         questionary.Choice("(3) Datasetleri ana dataset icine birlestir", "merge"),
         questionary.Choice("(4) Train/val/test oranlariyla yeniden bolustur", "split"),
         questionary.Choice("(5) Ana dataseti images/split + labels/split duzenine cevir", "convert"),
         questionary.Choice("(6) Datasetleri yalnizca kontrol et", "validate"),
         questionary.Choice("(7) Ana dataseti ZIP dosyasi yap", "zip"),
         questionary.Choice("(0) Cikis", "exit")],
    )
    {"empty_labels": create_empty_labels_for_images,
     "filter": filter_classes, "merge": merge_datasets,
     "split": redistribute_datasets, "convert": convert_to_zip_layout,
     "validate": validation_menu, "zip": create_dataset_zip,
     "exit": lambda: print("Cikis yapildi.")}[action]()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nIslem kullanici tarafindan iptal edildi.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nHATA: {exc}")
        raise SystemExit(1)
