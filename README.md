<!-- Proje-Resmi -->
<!-- ne kadar fotograf olmali, dosyalar nerede olmali zip icerisinde , zip i atma, ipynb deki bagimliliklari kendinden cek  vs  -->
<!-- kodun tr en olmasi olabilir-->
<!-- colab koduna collabda ac baglantisi koy-->

## 👀 cv-model-training-pt Overview  1/3  
<h1 align="center">Computer Vision AI model training for rasberry pi 5 and interactive command-line application for filtering, merging, splitting, validating, reorganizing, and packaging YOLO datasets</h1>  


## 🚀 Features
- [x] Create empty YOLO label files for background and negative images
- [x] Filter datasets using one or multiple selected classes
- [x] Keep selected classes or process all unselected classes
- [x] Copy filtered image-label pairs into a separate dataset
- [x] Delete selected image-label pairs after explicit confirmation
- [x] Limit the number of images separately for each class
- [x] Preserve the existing train, val, and test distribution during filtering
- [x] Choose whether images containing unselected objects are allowed
- [x] Automatically remove unwanted bounding boxes from copied labels
- [x] Merge multiple classes into a single class
- [x] Convert all label class IDs to a user-selected class
- [x] Automatically detect and display classes from data.yaml
- [x] Reindex remaining class IDs starting from 0
- [x] Automatically update data.yaml after class changes
- [x] Merge multiple YOLO datasets into one main dataset
- [x] Match classes automatically by their names in data.yaml
- [x] Remap source class IDs to the correct destination IDs
- [x] Preserve existing destination classes while adding new classes
- [x] Prevent class ID conflicts that could corrupt labels
- [x] Rename image-label pairs together when filenames conflict
- [x] Redistribute datasets using customizable train, val, and test percentages
- [x] Use an 80/10/10 train, val, and test split by default
- [x] Support datasets containing either single or multiple classes
- [x] Approximately preserve class distribution in multi-class datasets
- [x] Prioritize images containing rare classes during dataset splitting
- [x] Preserve empty labels used for negative images
- [x] Detect missing image-label pairs and offer repair options
- [x] Detect orphan label files that do not have matching images
- [x] Convert datasets to the images/split + labels/split directory structure
- [x] Support legacy valid directories and convert them to val
- [x] Validate YOLO detection and segmentation/polygon labels
- [x] Validate class IDs, coordinates, image-label counts, and duplicate filenames
- [x] Report image, label, bounding-box, negative-image, and per-class statistics
- [x] Automatically back up labels and data.yaml before modifying datasets
- [x] Display progress information during long-running operations
- [x] Browse and select directories across different disks and locations
- [x] Use the same interactive directory browser for every source and destination selection
- [x] Create ZIP archives containing only data.yaml, images, and labels
- [x] Preserve empty split directories inside generated ZIP archives
- [x] Verify ZIP integrity after archive creation
- [x] Windows and Linux support


`You can use `[`merge_yolo_datasets_EN.py`](https://github.com/FURK4NGG/cv-model-training-pt/blob/main/merge_yolo_datasets_EN.py)` to easily organize, filter, merge, validate, and redistribute your datasets before training.You can then use `[`Wildlife_YOLO11l_Hailo8_Training_EN.ipynb`](https://github.com/FURK4NGG/cv-model-training-pt/blob/main/Wildlife_YOLO11l_Hailo8_Training_EN.ipynb)` as a guide to learn the details of the model creation process. Once you understand this process, you can use `[`Universal_YOLO_Hailo_Training_EN.ipynb`](https://github.com/FURK4NGG/cv-model-training-pt/blob/main/Universal_YOLO_Hailo_Training_EN.ipynb)` to create well-trained models tailored to your specific needs.`


## 💡 Recommendations and Important Notes

Start with a main dataset directory containing a `data.yaml` file. This file should define the IDs and names of every class you plan to use.

When new datasets are merged into the main dataset, the `class_id` at the beginning of each copied label row is automatically remapped to the corresponding class ID defined in the main `data.yaml`.

Make sure that the class name in the source dataset’s `data.yaml` exactly matches the corresponding class name in the main dataset’s `data.yaml`. Otherwise, the program may interpret them as different classes.

For example:

```yaml
# Main dataset
names:
  0: bear
  1: boar
  2: deer
```

```yaml
# Source dataset
names:
  0: bear
```

When the source dataset is merged, its `bear` labels remain mapped to class ID `0`. If `bear` had a different ID in the source dataset, the copied label files would be remapped automatically.

A high-quality dataset should:

* Contain enough images for every class.
* Have a train/validation/test distribution close to `80/10/10`.
* Include different environments, lighting conditions, distances, angles, and object appearances.
* Avoid consecutive or nearly identical frames captured from the same video or camera-trap event.
* Keep images from the same event in the same split to prevent data leakage.
* Represent every important class in train, validation, and test whenever enough examples are available.
* Keep images from the same event in the same split to prevent data leakage.
    For example, do not place some of 20 consecutive images from the same camera-trap recording in train and others in val or test. All images from that event should remain in the same split. Otherwise, the model may encounter almost identical scenes during training and testing, producing misleadingly high evaluation results.

* Whenever enough examples are available, include every important class in train, validation, and test.
    For example, the bear class should not exist only in train. To properly measure whether the model can detect bears, val and test should also contain different bear images.


## Train, Validation, and Test Distribution

<details>
<summary>How should the data be distributed?</summary>

### TRAIN

Most of the available data should be placed in `train`.

For a bear detector, this should include:

- Different bear species and appearances.
- Daytime, nighttime, and infrared images.
- Close, distant, small, and large bears.
- Animals partially hidden behind trees or other objects.
- Different seasons and weather conditions.
- Different camera angles.
- Negative images such as empty forests.

For example, approximately 2,400 of 3,000 bear images can be placed in `train` when using an 80/10/10 split.

### VAL

The model does not learn its weights from `val` images. Validation data is used after or during training iterations to measure performance on images that are not part of the training split.

Validation results influence:

- Selection of the best checkpoint.
- Early stopping.
- Precision, recall, and mAP measurements.
- Detection of overfitting.
- Decisions about settings such as the confidence threshold.

For 3,000 images, approximately 300 images can be placed in `val`.

The validation split must also contain bear examples. However, they must not be copies of training images or neighboring frames from the same video event.

### TEST

The `test` split is used only for the final evaluation after training and model tuning have been completed.

If training parameters are repeatedly changed after reviewing test results, the test set is no longer impartial and effectively becomes a second validation set.

For 3,000 images, approximately 300 images can be placed in `test`.

### NEGATIVE IMAGES

Negative images are background images that do not contain any of the classes used by the model. Examples include empty forests, streets, roads, and empty camera-trap scenes.

Suggested starting ranges:

```text
Minimum:     10%
Recommended: 20%
Upper range: 25-30%
```

These percentages are not universal rules. The correct ratio depends on the real deployment environment and should be adjusted using validation results and false-positive behavior.

</details>

## 📦 Installation

### Windows

Install the required Python packages:

```powershell
py -m pip install questionary PyYAML
```

Run the program:

```powershell
py "merge_yolo_datasets.py"
```


The program asks you to select Windows or Linux mode when it starts.

### Linux Packages

<details>
<summary>Arch Linux packages</summary>

```bash
sudo pacman -S python xdg-utils
```

</details>

<details>
<summary>Debian/Ubuntu packages</summary>

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip xdg-utils
```

</details>

<details>
<summary>Fedora packages</summary>

```bash
sudo dnf install python3 python3-pip xdg-utils
```

</details>

### First Run on Linux

Using a virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install questionary PyYAML
python3 "merge_yolo_datasets.py"
```

### After the First Run

Activate the existing virtual environment and start the program:

```bash
source .venv/bin/activate
python3 "merge_yolo_datasets.py"
```


## 🔎 ALL APP FEATURES
<details>
<summary>For Nerds</summary>

## Automatic Dataset Discovery

The program automatically discovers dataset directories located in the directory from which it is executed.

An example project layout is:

```text
merge_yolo_datasets.py

bear/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Legacy datasets using `valid` instead of `val` can also be read. New standardized output directories are created using the name `val`.

Missing `train`, `val`, `test`, `images`, and `labels` directories are created when required by the selected operation.

## Folder Selection System

The same navigable folder picker is used for all source, destination, dataset, backup, and ZIP-output folder selections.

```text
Up/Down:           Move through the list
Right Arrow/Space: Enter the selected directory
Left Arrow/Esc:    Go to the parent directory
Enter:             Select the current directory
Ctrl+C:            Cancel the operation
```

The picker is not restricted to the directory from which the program was started. You can navigate between different disks and directories when selecting datasets, sources, destinations, and ZIP-output locations.

On Windows, available drives such as `C:\` and `D:\` are listed.

On Linux, you can navigate through accessible directories in the filesystem.

## Supported Working Dataset Layout

Dataset-editing operations use the following working layout:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The program can read a legacy `valid` directory, but standardized output uses `val`.

## 1. Create Empty/Negative Labels for Images

This option creates empty YOLO label files for background images that do not contain any target object.

First, select the actual `images` directory containing the photographs.

The program automatically identifies the sibling `labels` directory at the same level.

For example:

```text
train/
├── images/
└── labels/
```

An empty `.txt` file with the same stem is created for every image:

```text
forest001.jpg -> forest001.txt
forest002.png -> forest002.txt
```

The generated `.txt` files are completely empty. An empty label indicates that the corresponding image is a negative image and contains no target object.

If the `labels` directory does not exist, it is created automatically.

If the directory already contains label files, the program requests explicit confirmation:

```text
This labels directory is not empty. Do you want to overwrite the existing label files?
```

If the operation is confirmed, the existing `.txt` files are backed up first:

```text
labels_empty_backup_....zip
```

Only label files whose names correspond to images in the selected `images` directory are emptied. Unrelated label files are not modified.

If multiple images have the same stem, the operation stops. For example, having both `forest01.jpg` and `forest01.png` in the same directory is unsafe because both images would correspond to the same `forest01.txt` label file.

Progress is displayed while labels are created. When the operation finishes, the program verifies that an empty label exists for every selected image.

This feature must only be used for genuinely empty images. If an image contains an animal but receives an empty label, the model may learn to treat that animal as background.

A fixed negative-image ratio does not apply to every project. As a starting point, approximately 10-20% of the combined train, validation, and test images can be negative. Adjust this ratio according to the real camera-trap environment and validation results.

## 2. Filter/Reduce Classes

This menu contains three operations:

```text
(1) Keep or remove classes
(2) Merge multiple classes into one class
(3) Convert all label class IDs to one value
```

### 2.1. Keep or Remove Classes

One or more classes can be selected.

The prompts follow this order:

```text
Dataset whose classes will be filtered
Select the classes you want to filter
How should the images be filtered?
Which operation do you want to perform?
Which group should the operation apply to?
How many images should be filtered for each class?
```

There are two image-filtering methods.

#### Method 1: Keep selected boxes and remove unwanted boxes

```text
An object whose class will not exist in the new data.yaml may still appear in an image used to train the model for a selected class (recommended)
```

With this method:

- Bounding boxes belonging to selected classes are preserved.
- Bounding boxes belonging to unselected classes are removed from the generated label file.
- An image with no remaining target box can be kept as a negative image with an empty label.

However, an object that remains visible in an image after its annotation has been removed may be interpreted as background by the model. The resulting dataset should therefore be inspected visually.

#### Method 2: Use only images containing selected-class boxes

```text
Use only images that contain bounding boxes belonging to the selected classes
```

With this method, an image is not used if it contains a bounding box belonging to any unselected class.

This produces a cleaner class-specific dataset but may significantly reduce the number of usable images.

The operation can be applied to:

```text
Selected classes
Unselected classes
```

The program asks separately how many images should be processed for each affected class.

```text
0 = Use every eligible image
```

If the available distribution for a class after filtering is:

```text
train: 10
val:    4
test:   3
```

and eight images are requested, the program approximately preserves the split proportions:

```text
train: 5
val:   2
test:  1
```

Similarly, if the available distribution is `10000/4000/3000` and 800 images are requested, approximately `471/188/141` images are selected.

The files are selected using a fixed seed. Repeating the operation with the same dataset and settings should produce the same result whenever possible.

If one image contains multiple selected classes, it can contribute to multiple per-class quotas, but it is copied or deleted only once. Therefore, the total number of unique images may differ from the sum of the numbers entered for each class.

Available actions:

```text
Copy to another directory
Delete
```

#### Copy to Another Directory

First, select the destination directory.

The program then asks:

```text
Name of the new target dataset directory:
```

If this field is left empty, the selected directory itself is used as the target dataset. The required `train`, `val`, `test`, and `data.yaml` files and directories are created directly in that location, or existing ones are used.

If a name is entered, a dataset directory with that name is created under the selected directory, or an existing dataset with the same name is used.

Filtered images are copied while preserving their existing split:

```text
Source train -> Target train
Source val   -> Target val
Source test  -> Target test
```

The label file with the matching stem is copied together with each image.

If the destination already contains `data.yaml`, the source and destination class names are compared.

If the same class name exists in the destination under a different ID, the program offers to use the destination ID.

For example:

```text
Source:      0 = bear
Destination: 2 = bear
```

The `bear` ID in the copied label files is converted from `0` to `2`.

If the user rejects this mapping, the copy operation is cancelled to prevent dataset corruption.

If a source class ID is already used by another class in the destination, the program suggests the nearest available safe ID.

For example:

```text
Destination:
0 = bear
1 = boar

Source:
1 = wolf
```

The program asks whether the filtered `wolf` label values should be changed to `2`.

Existing classes and labels in the destination dataset are preserved. Only newly copied label files are remapped when necessary.

If label files exist in the destination but `data.yaml` is missing, the meaning of the existing IDs cannot be determined safely. The operation stops to prevent dataset corruption.

If a filename collision occurs, the image and label are renamed together:

```text
bear_dataset__image001.jpg
bear_dataset__image001.txt
```

Before copying, the destination dataset's current labels and YAML files are backed up.

#### Delete

This action removes eligible image-label pairs from the dataset.

If an image contains multiple classes, the program warns that annotations belonging to other classes may also be lost when the image is removed.

Deletion does not begin without explicit confirmation.

Automatic backups usually contain label and YAML files. Deleted image files cannot be recovered from these metadata-only backups.

### 2.2. Merge Multiple Classes into One Class

This option combines multiple classes in the same dataset under one class.

For example:

```text
polar_bear
black_bear
brown_bear
```

These classes can be merged into `bear`.

First, select at least two classes to merge.

Then select which of those classes will be used as the target class.

Every selected class ID is converted to the target class ID.

If an image contains multiple bounding boxes, the boxes are not deleted. Each box is preserved, but its class ID is changed to represent the target class.

After the merge, the remaining class IDs are reordered starting from `0`, and `data.yaml` is updated.

Empty negative label files remain empty.

Labels and `data.yaml` are backed up before the operation. Progress is displayed during processing, and the result is validated afterward.

### 2.3. Convert All Label Class IDs to One Value

This option changes the first value of every non-empty label row to one selected class ID.

If class names are available in `data.yaml`, the program lists the detected classes and allows the target class to be selected. The class name therefore does not need to be typed manually with an exact match.

If the program cannot find `data.yaml` or a usable class list, it reports:

```text
Class information could not be found. You must enter the target class ID and class name manually.
```

Only the first value of each label row is changed. Coordinates, detection boxes, and polygon points are preserved.

Empty negative label files remain empty.

After the operation, `data.yaml` is updated for the single target class.

For direct single-class YOLO training, the expected class ID is generally `0`. The program displays a warning if a different value is selected.

Labels and YAML files are backed up before the operation.

## 3. Merge Datasets into a Main Dataset

This option merges multiple YOLO datasets into one main dataset.

Source datasets are selected one by one using the folder picker. The selection order affects the order in which new classes are added to a newly created main dataset.

Class names are read from the `names` field in each dataset's `data.yaml`, not from directory names.

The program asks whether a new target dataset should be created or an existing main dataset should be used.

When creating a new dataset, select its parent directory and enter the name of the new dataset directory.

When using an existing dataset, the destination dataset can be selected from any accessible directory.

The recommended option matches classes by the names stored in `data.yaml`.

Example source dataset:

```yaml
names:
  0: polar_bear
  1: black_bear
  2: brown_bear
```

Example main dataset:

```yaml
names:
  0: bear
  1: boar
  2: deer
  3: wolf
  4: cow
  5: person
  6: dog
```

Merged result:

```yaml
names:
  0: bear
  1: boar
  2: deer
  3: wolf
  4: cow
  5: person
  6: dog
  7: polar_bear
  8: black_bear
  9: brown_bear
```

Source label IDs are converted automatically:

```text
0 -> 7
1 -> 8
2 -> 9
```

Source files are not modified. Class-ID conversion is applied only to label files copied into the main dataset.

If the same class name exists in both datasets, a new class is not created. The existing class ID in the target dataset is used.

For example, if both source and destination contain `brown_bear`, a second `brown_bear` class is not added.

Unnecessary capitalization and surrounding whitespace differences are normalized when class names are compared.

If a filename collision occurs, the image and label are renamed together:

```text
bear_dataset__image001.jpg
bear_dataset__image001.txt
```

Copying a multi-class dataset without changing its class IDs is safe only when the ID meanings in both datasets are identical:

```text
Source 0 = bear
Target 0 = bear
```

If the source uses `0=polar_bear` while the target uses `0=bear`, copying the labels directly is incorrect. For normal use, select the option that matches classes by their `data.yaml` names.

If the target dataset already contains `data.yaml`, the program may offer to preserve the current class list or rebuild it according to source-selection order. If classes are reordered, existing target labels are also remapped by class name.

The target dataset's labels and YAML files are backed up before merging.

Progress is displayed during copying and class-ID conversion. The main dataset is validated when the operation finishes.

## 4. Redistribute Using Train/Val/Test Ratios

This option redistributes image-label pairs from one or more datasets across train, validation, and test directories.

The program asks:

```text
How many dataset directories do you have?
Select dataset directories containing one class
Select dataset directories containing multiple classes
Train percentage
Validation percentage
Test percentage
```

The first number is the number of dataset directories to process, not the number of classes.

The total number of selected single-class and multi-class datasets must match the entered dataset count.

Default ratios:

```text
train: 80%
val:   10%
test:  10%
```

The percentages must total `100%`.

A split can be assigned `0%`. Its directory is still created but remains empty.

If `val` or `test` does not exist in a source dataset, the required directories are created automatically.

The program collects all image-label pairs from the current train, validation, and test directories and redistributes them.

For single-class datasets, distribution is based on the number of images.

For multi-class datasets, label contents are read and the existing class distribution is preserved as closely as possible.

For example, if the existing bounding-box distribution is:

```text
polar_bear : black_bear : brown_bear
3          : 5          : 8
```

the program does not force an equal number of images from each class. It attempts to preserve the approximate `3:5:8` distribution.

Images containing rare classes receive priority during placement.

If one image contains three classes, that image contributes toward the target of all three classes. Exact per-class mathematical targets are therefore not always possible in multi-label datasets.

The total train, validation, and test image counts are still distributed according to the requested ratios.

Images in train, validation, and test must be different. The same image must not appear in more than one split.

It is not sufficient for a class to exist only in train. Whenever enough examples are available, every trained class, such as `bear`, should also be represented in validation and test.

`train` is used to learn model weights.

`val` is used during training to measure performance on unseen images and evaluate training decisions.

`test` is used after training and tuning are complete to obtain an impartial final performance measurement.

Highly similar frames from the same video or camera-trap event should not be distributed across different splits. Otherwise, the model may have already seen nearly identical views of the event, causing evaluation results to appear better than real-world performance.

The program cannot automatically determine whether visually similar frames originate from the same video or camera-trap event. Such frames must be grouped by event before redistribution.

Progress is shown while files are moved to temporary storage and placed into their new splits.

Labels and `data.yaml` are backed up before redistribution. All images are not included in the automatic backup because that could create unnecessarily large archives containing hundreds of gigabytes.

### Missing Label Handling

If an image does not have a label file with the same stem, the program reports:

```text
The label file belonging to image001.jpg could not be found
```

It then offers:

```text
(1) Open the image
(2) Continue / skip for now
(3) Terminate the program
```

After opening the image:

```text
(1) Delete the image
(2) Create an empty label
(3) Do nothing and terminate the program
```

The empty-label option is only appropriate for genuine negative images.

Standalone `.txt` label files with no matching image are not deleted or modified automatically. The program reports an error and displays their locations.

On Windows, images are opened using the operating system's default image viewer.

On Linux, `xdg-open` is used. The program may request installation of `xdg-utils` when necessary.

## 5. Convert the Main Dataset to `images/split + labels/split`

This option converts the main dataset from the working layout to the final layout used for training, sharing, or ZIP packaging.

Source layout:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Converted layout:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

If a legacy `valid` directory is found, it is converted to the standardized `val` name.

Files are moved into the new layout rather than copied.

If `images` or `labels` already exists at the dataset root, the operation stops to prevent accidental overwriting.

The train, validation, and test paths in `data.yaml` are updated for the new folder layout.

A backup of `data.yaml` is stored next to the dataset before conversion.

After conversion, the program validates the new directory layout and all image-label matches.

This should be treated as a final packaging step. Filtering, merging, and redistribution should be performed in the working layout before conversion.

## 6. Validate Datasets Only

This option validates one or more datasets without modifying any files.

Datasets can be selected from different disks and directories.

Progress is displayed during validation.

The program checks:

```text
Does every image have a .txt label file with the same stem?
Does every label have a corresponding image?
Are there multiple images with the same stem?
Is any class ID negative?
Is every class ID defined in data.yaml?
Are coordinates within the 0-1 range?
Are detection rows valid?
Are polygon rows valid?
Are bounding-box width and height values positive?
Are empty negative labels preserved?
Are image and label counts equal in each split?
How many bounding boxes exist for each class?
```

A normal object-detection row must contain a class ID followed by four coordinate values.

A polygon row must contain a class ID followed by a valid even number of coordinate values.

Example validation output:

```text
train: images=7000, labels=7000, boxes=8450, negatives=900 [OK]
val  : images=1000, labels=1000, boxes=1210, negatives=125 [OK]
test : images=1000, labels=1000, boxes=1195, negatives=125 [OK]
```

The number of images does not need to equal the number of bounding boxes.

An image can contain zero, one, or multiple bounding boxes.

The values that must match are the number of images and the number of corresponding label files.

## 7. Create a ZIP File from the Main Dataset

This option creates a ZIP archive from a main dataset that has already been converted to the final directory layout.

First, select the dataset to archive.

Then select the directory in which the ZIP file will be saved. The ZIP file does not have to be stored next to the source dataset.

Only the required dataset contents are added to the archive:

```text
data.yaml
images/
labels/
```

An unnecessary outer dataset directory is not added inside the ZIP.

Opening the ZIP therefore shows this structure directly:

```text
data.yaml
images/train/
images/val/
images/test/
labels/train/
labels/val/
labels/test/
```

Empty train, validation, or test directories are preserved.

ZIP64 support is used for large datasets.

Progress is displayed according to the number of files and amount of processed data.

The archive is first written to a temporary `.zip.part` file. It is renamed to the final `.zip` name only after the operation finishes successfully.

If a ZIP with the same name already exists, explicit confirmation is required before it is overwritten.

After creation, the program performs an archive-integrity test and verifies that all required directories are present.

## Final Validation

After operations such as filtering, class merging, dataset merging, redistribution, and folder conversion, the resulting dataset is validated automatically.

If a missing image, missing label, invalid class ID, or invalid coordinate is found, its location is displayed.

Even when automatic validation succeeds, visually inspect validation and test images before training.

Important checks include:

```text
Does the same image appear in multiple splits?
Were highly similar frames from the same event distributed across different splits?
Is every important class sufficiently represented in validation and test?
Are negative images genuinely free of all target objects?
Did filtering leave visible but unannotated target objects in any image?
```

Automatic validation can verify file structures and annotation values, but it cannot determine whether a visible object should have been annotated. Visual inspection remains necessary.

## Backups

Automatic backups are created before operations that modify data.

Example backup names:

```text
bear_filter_backup_....zip
bear_filter_copy_backup_....zip
bear_class_merge_backup_....zip
bear_class_id_backup_....zip
bear_merge_backup_....zip
bear_split_backup_....zip
labels_empty_backup_....zip
```

Backups are stored next to the relevant dataset.

Filtering, class merging, dataset merging, and redistribution backups usually contain label files and `data.yaml`.

All images are not included. This prevents extremely large and unnecessary backup archives from being created for large datasets.

Automatic metadata backups cannot restore deleted image files.

## Important Safety Notes

- Keep an independent copy of the original dataset before using deletion or move-based conversion operations.
- Do not assume that an automatic backup contains image files; most automatic backups protect labels and YAML metadata only.
- Do not manually merge datasets solely by matching numeric class IDs unless the class meaning of every ID is identical.
- Do not create empty labels for images that contain a target object.
- Do not repeatedly tune the model using test results; use validation results for tuning and reserve test data for final evaluation.
- Do not distribute neighboring frames from the same video or camera-trap event across different splits.
- Run validation before training and again before creating the final ZIP archive.

## Platform Notes

The program's folder selection, path creation, dataset processing, and image-opening behavior account for both Windows and Linux path structures.

On Linux, opening an image externally requires `xdg-open`, provided by the `xdg-utils` package.

On Windows, the default system image viewer is used.
</details>

<details>
<summary>Bilgi Hastaları için</summary>
    
## Otomatik Dataset Bulma

Program, çalıştırıldığı dizinde bulunan dataset klasörlerini otomatik olarak bulur.

Örnek proje düzeni:

```text
merge_yolo_datasets.py

bear/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Program, `val` yerine `valid` kullanan eski datasetleri de okuyabilir. Yeni ve standartlaştırılmış çıktı klasörleri `val` adıyla oluşturulur.

Eksik `train`, `val`, `test`, `images` ve `labels` klasörleri, seçilen işlem için gerekli olduklarında otomatik olarak oluşturulur.

## Klasör Seçme Sistemi

Programdaki bütün kaynak, hedef, dataset, yedekleme ve ZIP kayıt klasörü seçimlerinde aynı gezilebilir klasör seçici kullanılır.

```text
Yukarı/Aşağı:        Listede hareket et
Sağ Ok/Space:        Seçili klasöre gir
Sol Ok/Esc:          Bir üst klasöre çık
Enter:               Bulunduğun klasörü seç
Ctrl+C:              İşlemi iptal et
```

Klasör seçici, programın çalıştırıldığı dizinle sınırlı değildir. Dataset, kaynak, hedef ve ZIP kayıt konumlarını seçerken farklı diskler ve dizinler arasında gezebilirsiniz.

Windows’ta `C:\` ve `D:\` gibi kullanılabilir diskler listelenir.

Linux’ta dosya sistemindeki erişilebilir klasörler arasında gezilebilir.

## Desteklenen Çalışma Dataset Düzeni

Dataset düzenleme işlemlerinde aşağıdaki çalışma düzeni kullanılır:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Program eski datasetlerdeki `valid` klasörünü okuyabilir ancak standartlaştırılmış çıktılarda `val` kullanılır.

## 1. Fotoğraflar İçin Boş/Negatif Label Oluştur

Bu seçenek, hedef objelerin hiçbirini içermeyen arka plan görüntüleri için boş YOLO label dosyaları oluşturur.

Öncelikle fotoğrafların bulunduğu gerçek `images` klasörü seçilir.

Program aynı seviyede bulunan `labels` klasörünü otomatik olarak belirler.

Örneğin:

```text
train/
├── images/
└── labels/
```

Her görüntü için aynı kök ada sahip boş bir `.txt` dosyası oluşturulur:

```text
forest001.jpg -> forest001.txt
forest002.png -> forest002.txt
```

Oluşturulan `.txt` dosyaları tamamen boş olur. Boş label, karşılık gelen görüntünün negatif olduğunu ve herhangi bir hedef obje içermediğini belirtir.

`labels` klasörü yoksa otomatik olarak oluşturulur.

Klasörde daha önceden oluşturulmuş label dosyaları varsa program açık onay ister:

```text
Bu labels klasörü boş değil. Mevcut label dosyalarının üzerine yazmak istiyor musunuz?
```

İşlem onaylanırsa mevcut `.txt` dosyaları önce yedeklenir:

```text
labels_empty_backup_....zip
```

Yalnızca seçilen `images` klasöründeki görüntülerle aynı ada sahip label dosyaları boşaltılır. İlgisiz label dosyaları değiştirilmez.

Aynı kök ada sahip birden fazla görüntü bulunursa işlem durdurulur. Örneğin aynı klasörde hem `forest01.jpg` hem de `forest01.png` bulunması güvenli değildir çünkü iki görüntü de aynı `forest01.txt` label dosyasına karşılık gelir.

Label dosyaları oluşturulurken ilerleme durumu gösterilir. İşlem tamamlandığında program seçilen her görüntü için boş bir label bulunup bulunmadığını doğrular.

Bu özellik yalnızca gerçekten boş görüntüler için kullanılmalıdır. Bir görüntüde hayvan bulunduğu hâlde boş label oluşturulursa model bu hayvanı arka plan olarak öğrenebilir.

Her proje için geçerli sabit bir negatif görüntü oranı yoktur. Başlangıç olarak toplam train, validation ve test görüntülerinin yaklaşık `%10–20` kadarı negatif olabilir. Bu oran gerçek fotokapan ortamına ve doğrulama sonuçlarına göre ayarlanmalıdır.

## 2. Class’ları Filtrele/Azalt

Bu menü üç işlem içerir:

```text
(1) Class’ları tut veya kaldır
(2) Birden fazla class’ı tek class’ta birleştir
(3) Bütün label class ID’lerini tek bir değere dönüştür
```

### 2.1. Class’ları Tut veya Kaldır

Bir veya birden fazla class seçilebilir.

Sorular şu sırayla gösterilir:

```text
Class’ları filtrelenecek dataset
Filtrelemek istediğiniz class’ları seçin
Resimler nasıl filtrelenmeli?
Hangi işlemi yapmak istiyorsunuz?
İşlem hangi gruba uygulanmalı?
Her class için kaç görüntü filtrelenmeli?
```

İki farklı görüntü filtreleme yöntemi bulunur.

#### Yöntem 1: Seçilen kutuları koru ve istenmeyen kutuları kaldır

```text
Yeni data.yaml içinde bulunmayacak bir class’a ait obje, modeli seçilen class için eğiten bir görüntüde bulunabilir (önerilen)
```

Bu yöntemde:

* Seçilen class’lara ait bounding box’lar korunur.
* Seçilmeyen class’lara ait bounding box’lar oluşturulan label dosyasından kaldırılır.
* Hedef kutusu kalmayan bir görüntü, boş label ile negatif görüntü olarak tutulabilir.

Ancak annotation’ı kaldırıldıktan sonra görüntüde görünmeye devam eden bir obje model tarafından arka plan olarak değerlendirilebilir. Bu nedenle oluşturulan dataset görsel olarak kontrol edilmelidir.

#### Yöntem 2: Yalnızca seçilen class kutularını içeren görüntüleri kullan

```text
Yalnızca seçilen class’lara ait bounding box içeren görüntüleri kullan
```

Bu yöntemde, görüntüde seçilmeyen herhangi bir class’a ait bounding box bulunuyorsa görüntü kullanılmaz.

Bu yöntem daha temiz bir class dataseti oluşturur ancak kullanılabilir görüntü sayısını önemli ölçüde azaltabilir.

İşlem şu gruplardan birine uygulanabilir:

```text
Seçilen class’lar
Seçilmeyen class’lar
```

Program etkilenen her class için kaç görüntü işleneceğini ayrı ayrı sorar.

```text
0 = Uygun olan bütün görüntüleri kullan
```

Filtreleme sonrasında bir class için kullanılabilir dağılım şu şekildeyse:

```text
train: 10
val:    4
test:   3
```

ve sekiz görüntü istenirse program split oranlarını yaklaşık olarak korur:

```text
train: 5
val:   2
test:  1
```

Benzer şekilde kullanılabilir dağılım `10000/4000/3000` ise ve 800 görüntü istenirse yaklaşık `471/188/141` görüntü seçilir.

Dosyalar sabit bir seed kullanılarak seçilir. Aynı dataset ve aynı ayarlarla işlem tekrarlandığında mümkün olduğunca aynı sonuç elde edilir.

Bir görüntü birden fazla seçilmiş class içeriyorsa birden fazla class kotasına katkı sağlayabilir ancak yalnızca bir kez kopyalanır veya silinir. Bu nedenle toplam benzersiz görüntü sayısı, class başına girilen sayıların toplamından farklı olabilir.

Kullanılabilir işlemler:

```text
Başka bir dizine kopyala
Sil
```

#### Başka Bir Dizine Kopyalama

Öncelikle hedef dizin seçilir.

Program daha sonra şunu sorar:

```text
Yeni hedef dataset klasörünün adı:
```

Bu alan boş bırakılırsa seçilen dizinin kendisi hedef dataset olarak kullanılır. Gerekli `train`, `val`, `test` ve `data.yaml` dosya ve klasörleri doğrudan bu konumda oluşturulur veya mevcut olanlar kullanılır.

Bir isim girilirse seçilen dizinin altında bu isimde bir dataset klasörü oluşturulur veya aynı isimdeki mevcut dataset kullanılır.

Filtrelenmiş görüntüler mevcut split’leri korunarak kopyalanır:

```text
Kaynak train -> Hedef train
Kaynak val   -> Hedef val
Kaynak test  -> Hedef test
```

Her görüntüyle birlikte aynı kök ada sahip label dosyası da kopyalanır.

Hedefte `data.yaml` bulunuyorsa kaynak ve hedef class isimleri karşılaştırılır.

Aynı class ismi hedefte farklı bir ID altında bulunuyorsa program hedefteki ID’yi kullanmayı teklif eder.

Örneğin:

```text
Kaynak: 0 = bear
Hedef:  2 = bear
```

Kopyalanan label dosyalarındaki `bear` ID’si `0` değerinden `2` değerine dönüştürülür.

Kullanıcı bu eşleştirmeyi reddederse datasetin bozulmasını önlemek için kopyalama işlemi iptal edilir.

Kaynak class ID’si hedefte başka bir class tarafından kullanılıyorsa program en yakın kullanılabilir ve güvenli ID’yi önerir.

Örneğin:

```text
Hedef:
0 = bear
1 = boar

Kaynak:
1 = wolf
```

Program filtrelenen `wolf` label değerlerini `2` yapmak isteyip istemediğinizi sorar.

Hedef datasetteki mevcut class’lar ve label dosyaları korunur. Yalnızca yeni kopyalanan label dosyaları gerekli olduğunda yeniden eşleştirilir.

Hedefte label dosyaları bulunduğu hâlde `data.yaml` yoksa mevcut ID’lerin hangi class anlamına geldiği güvenli biçimde belirlenemez. Datasetin bozulmasını önlemek için işlem durdurulur.

Dosya adı çakışırsa görüntü ve label birlikte yeniden adlandırılır:

```text
bear_dataset__image001.jpg
bear_dataset__image001.txt
```

Kopyalama öncesinde hedef datasetin mevcut label ve YAML dosyaları yedeklenir.

#### Silme

Bu işlem uygun bulunan görüntü-label çiftlerini datasetten kaldırır.

Bir görüntü birden fazla class içeriyorsa program diğer class’lara ait annotation’ların da görüntüyle birlikte kaybedilebileceğini bildirir.

Silme işlemi açık onay alınmadan başlamaz.

Otomatik yedekler çoğunlukla label ve YAML dosyalarını içerir. Silinen görüntü dosyaları yalnızca metadata içeren bu yedeklerden geri getirilemez.

### 2.2. Birden Fazla Class’ı Tek Class’ta Birleştir

Bu seçenek aynı dataset içerisindeki birden fazla class’ı tek bir class altında birleştirir.

Örneğin:

```text
polar_bear
black_bear
brown_bear
```

Bu class’lar `bear` altında birleştirilebilir.

Öncelikle birleştirilecek en az iki class seçilir.

Daha sonra seçilen class’lardan hangisinin hedef class olarak kullanılacağı belirlenir.

Seçilen bütün class ID’leri hedef class ID’sine dönüştürülür.

Bir görüntüde birden fazla bounding box bulunuyorsa kutular silinmez. Her kutu korunur ancak class ID’si hedef class’ı gösterecek şekilde değiştirilir.

Birleştirme sonrasında kalan class ID’leri `0` değerinden başlayarak yeniden sıralanır ve `data.yaml` güncellenir.

Boş negatif label dosyaları boş kalır.

İşlem öncesinde label dosyaları ve `data.yaml` yedeklenir. İşlem sırasında ilerleme gösterilir ve sonuç daha sonra doğrulanır.

### 2.3. Bütün Label Class ID’lerini Tek Bir Değere Dönüştür

Bu seçenek bütün dolu label satırlarının ilk değerini seçilen tek bir class ID’sine dönüştürür.

`data.yaml` içinde class isimleri bulunuyorsa program algıladığı class’ları listeler ve hedef class’ın seçilmesini sağlar. Böylece class adının birebir elle yazılması gerekmez.

Program `data.yaml` veya kullanılabilir bir class listesi bulamazsa şu mesajı gösterir:

```text
Class bilgisi bulunamadı. Hedef class ID’sini ve class adını elle girmeniz gerekiyor.
```

Yalnızca her label satırının ilk değeri değiştirilir. Koordinatlar, detection kutuları ve polygon noktaları korunur.

Boş negatif label dosyaları boş kalır.

İşlem sonrasında `data.yaml` tek hedef class’a göre güncellenir.

Doğrudan tek class’lı YOLO eğitimi için beklenen class ID genellikle `0` değeridir. Farklı bir değer seçilirse program uyarı gösterir.

İşlem öncesinde label ve YAML dosyaları yedeklenir.

## 3. Datasetleri Ana Dataset İçinde Birleştir

Bu seçenek birden fazla YOLO datasetini tek bir ana dataset içinde birleştirir.

Kaynak datasetler klasör seçici kullanılarak sırayla seçilir. Seçim sırası, yeni oluşturulan ana datasete yeni class’ların hangi sırayla ekleneceğini etkiler.

Class isimleri klasör adlarından değil, her datasetin `data.yaml` dosyasındaki `names` alanından okunur.

Program yeni bir hedef dataset oluşturulmasını mı yoksa mevcut bir ana datasetin kullanılmasını mı istediğinizi sorar.

Yeni dataset oluşturulurken üst dizin seçilir ve yeni dataset klasörünün adı girilir.

Mevcut dataset kullanılacaksa hedef dataset erişilebilir herhangi bir dizinden seçilebilir.

Önerilen seçenek, class’ları `data.yaml` içinde kayıtlı isimlerine göre eşleştirir.

Örnek kaynak dataset:

```yaml
names:
  0: polar_bear
  1: black_bear
  2: brown_bear
```

Örnek ana dataset:

```yaml
names:
  0: bear
  1: boar
  2: deer
  3: wolf
  4: cow
  5: person
  6: dog
```

Birleştirme sonucu:

```yaml
names:
  0: bear
  1: boar
  2: deer
  3: wolf
  4: cow
  5: person
  6: dog
  7: polar_bear
  8: black_bear
  9: brown_bear
```

Kaynak label ID’leri otomatik olarak dönüştürülür:

```text
0 -> 7
1 -> 8
2 -> 9
```

Kaynak dosyalar değiştirilmez. Class ID dönüşümü yalnızca ana datasete kopyalanan label dosyalarına uygulanır.

Aynı class ismi iki datasette de bulunuyorsa yeni bir class oluşturulmaz. Hedef datasetteki mevcut class ID kullanılır.

Örneğin hem kaynakta hem de hedefte `brown_bear` varsa ikinci bir `brown_bear` class’ı eklenmez.

Class isimleri karşılaştırılırken gereksiz büyük-küçük harf ve baştaki veya sondaki boşluk farklılıkları normalize edilir.

Dosya adı çakışırsa görüntü ve label birlikte yeniden adlandırılır:

```text
bear_dataset__image001.jpg
bear_dataset__image001.txt
```

Çok class’lı bir dataseti class ID’lerini değiştirmeden kopyalamak yalnızca iki datasetteki ID anlamları tamamen aynıysa güvenlidir:

```text
Kaynak 0 = bear
Hedef  0 = bear
```

Kaynakta `0=polar_bear`, hedefte `0=bear` ise label dosyalarını doğrudan kopyalamak yanlıştır. Normal kullanımda class’ları `data.yaml` isimlerine göre eşleştiren seçenek kullanılmalıdır.

Hedef datasette zaten `data.yaml` bulunuyorsa program mevcut class listesini korumayı veya kaynak seçim sırasına göre yeniden oluşturmayı teklif edebilir. Class’lar yeniden sıralanırsa hedefteki mevcut label dosyaları da class isimlerine göre yeniden eşleştirilir.

Birleştirme öncesinde hedef datasetin label ve YAML dosyaları yedeklenir.

Kopyalama ve class ID dönüştürme sırasında ilerleme gösterilir. İşlem tamamlandığında ana dataset doğrulanır.

## 4. Train/Val/Test Oranlarıyla Yeniden Bölüştür

Bu seçenek bir veya birden fazla datasetteki görüntü-label çiftlerini train, validation ve test klasörleri arasında yeniden dağıtır.

Program şunları sorar:

```text
Kaç dataset klasörünüz var?
Tek class içeren dataset klasörlerini seçin
Birden fazla class içeren dataset klasörlerini seçin
Train yüzdesi
Validation yüzdesi
Test yüzdesi
```

İlk girilen sayı class sayısı değil, işlenecek dataset klasörü sayısıdır.

Seçilen tek class’lı ve çok class’lı datasetlerin toplam sayısı, girilen dataset sayısıyla aynı olmalıdır.

Varsayılan oranlar:

```text
train: %80
val:   %10
test:  %10
```

Yüzdelerin toplamı `%100` olmalıdır.

Bir split’e `%0` verilebilir. Bu durumda ilgili klasör yine oluşturulur ancak boş kalır.

Kaynak datasette `val` veya `test` bulunmasa bile gerekli klasörler otomatik olarak oluşturulur.

Program mevcut train, validation ve test klasörlerindeki bütün görüntü-label çiftlerini toplar ve yeniden dağıtır.

Tek class’lı datasetlerde dağıtım görüntü sayısına göre yapılır.

Çok class’lı datasetlerde label içerikleri okunur ve mevcut class dağılımı mümkün olduğunca korunur.

Örneğin mevcut bounding box dağılımı şu şekildeyse:

```text
polar_bear : black_bear : brown_bear
3          : 5          : 8
```

program her class’tan zorla eşit sayıda görüntü seçmez. Yaklaşık `3:5:8` dağılımını korumaya çalışır.

Nadir class içeren görüntülere yerleştirme sırasında öncelik verilir.

Bir görüntü üç class içeriyorsa bu görüntü üç class’ın da hedefine katkı sağlar. Bu nedenle çok etiketli datasetlerde class başına matematiksel hedeflerin tamamen karşılanması her zaman mümkün değildir.

Toplam train, validation ve test görüntü sayıları yine belirlenen oranlara göre dağıtılır.

Train, validation ve test içindeki görüntüler birbirinden farklı olmalıdır. Aynı görüntü birden fazla split içinde bulunmamalıdır.

Bir class’ın yalnızca train içinde bulunması yeterli değildir. Yeterli örnek varsa `bear` gibi eğitilen her class validation ve test içinde de temsil edilmelidir.

`train`, model ağırlıklarını öğrenmek için kullanılır.

`val`, eğitim sırasında modelin görmediği görüntüler üzerindeki başarısını ölçmek ve eğitim kararlarını değerlendirmek için kullanılır.

`test`, eğitim ve ayarlama tamamlandıktan sonra tarafsız bir son performans ölçümü elde etmek için kullanılır.

Aynı video veya fotokapan olayından gelen çok benzer kareler farklı split’lere dağıtılmamalıdır. Aksi hâlde model aynı olayın neredeyse aynı görüntülerini daha önce görmüş olabilir ve değerlendirme sonuçları gerçek kullanım başarısından daha iyi görünebilir.

Program, görsel olarak benzer karelerin aynı video veya fotokapan olayından gelip gelmediğini otomatik olarak belirleyemez. Bu karelerin yeniden bölüştürme işleminden önce olay bazında gruplandırılması gerekir.

Dosyalar geçici alana taşınırken ve yeni split’lere yerleştirilirken ilerleme durumu gösterilir.

Yeniden bölüştürme öncesinde label dosyaları ve `data.yaml` yedeklenir. Yüzlerce GB boyutunda gereksiz arşivler oluşmasını önlemek için bütün görüntüler otomatik yedeğe eklenmez.

### Eksik Label İşleme

Bir görüntünün aynı kök ada sahip label dosyası bulunamazsa program şu mesajı gösterir:

```text
image001.jpg dosyasına ait label dosyası bulunamadı
```

Daha sonra şu seçenekler sunulur:

```text
(1) Görüntüyü aç
(2) Devam et / şimdilik atla
(3) Programı sonlandır
```

Görüntü açıldıktan sonra:

```text
(1) Görüntüyü sil
(2) Boş label oluştur
(3) Hiçbir şey yapma ve programı sonlandır
```

Boş label oluşturma seçeneği yalnızca gerçekten negatif olan görüntüler için uygundur.

Karşılık gelen görüntüsü bulunmayan bağımsız `.txt` label dosyaları otomatik olarak silinmez veya değiştirilmez. Program hata göstererek bu dosyaların konumlarını bildirir.

Windows’ta görüntüler işletim sisteminin varsayılan görüntüleyicisiyle açılır.

Linux’ta `xdg-open` kullanılır. Gerekirse program `xdg-utils` paketinin kurulmasını isteyebilir.

## 5. Ana Dataseti `images/split + labels/split` Düzenine Dönüştür

Bu seçenek ana dataseti çalışma düzeninden eğitim, paylaşım veya ZIP paketleme için kullanılan son klasör düzenine dönüştürür.

Kaynak düzen:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Dönüştürülen düzen:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Eski bir `valid` klasörü bulunursa standart `val` adına dönüştürülür.

Dosyalar kopyalanmak yerine yeni düzene taşınır.

Dataset kökünde daha önceden `images` veya `labels` klasörü bulunuyorsa yanlışlıkla üzerine yazmayı önlemek için işlem durdurulur.

`data.yaml` içindeki train, validation ve test yolları yeni klasör düzenine göre güncellenir.

Dönüştürme öncesinde `data.yaml` dosyasının yedeği datasetin yanına kaydedilir.

İşlemden sonra program yeni klasör düzenini ve bütün görüntü-label eşleşmelerini doğrular.

Bu işlem son paketleme adımı olarak düşünülmelidir. Filtreleme, birleştirme ve yeniden bölüştürme işlemleri çalışma düzenindeyken yapılmalı, klasör dönüşümü bunlardan sonra uygulanmalıdır.

## 6. Datasetleri Yalnızca Kontrol Et

Bu seçenek herhangi bir dosyayı değiştirmeden bir veya birden fazla dataseti doğrular.

Datasetler farklı disklerden ve klasörlerden seçilebilir.

Doğrulama sırasında ilerleme durumu gösterilir.

Program şu kontrolleri yapar:

```text
Her görüntünün aynı kök ada sahip bir .txt label dosyası var mı?
Her label dosyasının karşılık gelen görüntüsü var mı?
Aynı kök ada sahip birden fazla görüntü var mı?
Herhangi bir class ID negatif mi?
Bütün class ID’leri data.yaml içinde tanımlı mı?
Koordinatlar 0–1 aralığında mı?
Detection satırları geçerli mi?
Polygon satırları geçerli mi?
Bounding box genişlik ve yükseklik değerleri pozitif mi?
Boş negatif label dosyaları korunmuş mu?
Her split içindeki görüntü ve label sayıları eşit mi?
Her class için kaç bounding box var?
```

Normal bir object detection satırında class ID’den sonra dört koordinat değeri bulunmalıdır.

Bir polygon satırında class ID’den sonra geçerli sayıda ve çiftler hâlinde koordinat değeri bulunmalıdır.

Örnek doğrulama çıktısı:

```text
train: görüntü=7000, label=7000, kutu=8450, negatif=900 [OK]
val  : görüntü=1000, label=1000, kutu=1210, negatif=125 [OK]
test : görüntü=1000, label=1000, kutu=1195, negatif=125 [OK]
```

Görüntü sayısıyla bounding box sayısının eşit olması gerekmez.

Bir görüntü sıfır, bir veya birden fazla bounding box içerebilir.

Eşit olması gereken değerler görüntü sayısı ile karşılık gelen label dosyası sayısıdır.

## 7. Ana Datasetten ZIP Dosyası Oluştur

Bu seçenek, son klasör düzenine dönüştürülmüş ana datasetten ZIP arşivi oluşturur.

Öncelikle arşivlenecek dataset seçilir.

Ardından ZIP dosyasının kaydedileceği klasör seçilir. ZIP dosyasının kaynak datasetin yanında bulunması gerekmez.

Arşive yalnızca gerekli dataset içerikleri eklenir:

```text
data.yaml
images/
labels/
```

ZIP içinde gereksiz bir dış dataset klasörü oluşturulmaz.

Bu nedenle ZIP açıldığında doğrudan şu yapı görülür:

```text
data.yaml
images/train/
images/val/
images/test/
labels/train/
labels/val/
labels/test/
```

Boş train, validation veya test klasörleri korunur.

Büyük datasetler için ZIP64 desteği kullanılır.

Dosya sayısına ve işlenen veri miktarına göre ilerleme durumu gösterilir.

Arşiv önce geçici bir `.zip.part` dosyasına yazılır. Yalnızca işlem başarıyla tamamlandıktan sonra son `.zip` adına dönüştürülür.

Aynı isimde bir ZIP dosyası bulunuyorsa üzerine yazılmadan önce açık onay istenir.

ZIP oluşturulduktan sonra program arşiv bütünlük testi yapar ve gerekli klasörlerin arşiv içinde bulunup bulunmadığını doğrular.

## Son Doğrulama

Filtreleme, class birleştirme, dataset birleştirme, yeniden bölüştürme ve klasör dönüşümü gibi işlemlerden sonra sonuç dataset otomatik olarak doğrulanır.

Eksik görüntü, eksik label, geçersiz class ID veya geçersiz koordinat bulunursa konumu gösterilir.

Otomatik doğrulama başarılı olsa bile eğitimden önce validation ve test görüntüleri görsel olarak incelenmelidir.

Kontrol edilmesi gereken önemli noktalar:

```text
Aynı görüntü birden fazla split içinde bulunuyor mu?
Aynı olaydan gelen çok benzer kareler farklı split’lere dağıtılmış mı?
Her önemli class validation ve test içinde yeterince temsil ediliyor mu?
Negatif görüntüler gerçekten bütün hedef objelerden arındırılmış mı?
Filtreleme sonrasında görüntüde görünen fakat annotation’ı bulunmayan hedef objeler kalmış mı?
```

Otomatik doğrulama dosya yapılarını ve annotation değerlerini kontrol edebilir ancak görüntüde görünen bir objenin etiketlenmiş olması gerekip gerekmediğini belirleyemez. Görsel kontrol yapılması hâlâ gereklidir.

## Yedeklemeler

Verileri değiştiren işlemlerden önce otomatik yedekler oluşturulur.

Örnek yedek adları:

```text
bear_filter_backup_....zip
bear_filter_copy_backup_....zip
bear_class_merge_backup_....zip
bear_class_id_backup_....zip
bear_merge_backup_....zip
bear_split_backup_....zip
labels_empty_backup_....zip
```

Yedekler ilgili datasetin yanına kaydedilir.

Filtreleme, class birleştirme, dataset birleştirme ve yeniden bölüştürme yedekleri çoğunlukla label dosyalarıyla `data.yaml` dosyasını içerir.

Bütün görüntüler yedeğe dahil edilmez. Bu sayede büyük datasetlerde aşırı büyük ve gereksiz yedek arşivlerinin oluşması önlenir.

Otomatik metadata yedekleri silinen görüntü dosyalarını geri getiremez.

## Önemli Güvenlik Notları

* Silme veya dosyaları taşıyan klasör dönüşümü işlemlerini kullanmadan önce orijinal datasetin bağımsız bir kopyasını saklayın.
* Otomatik yedeklerin görüntü dosyalarını içerdiğini varsaymayın. Çoğu otomatik yedek yalnızca label ve YAML metadata dosyalarını korur.
* Her ID’nin class anlamı tamamen aynı değilse datasetleri yalnızca sayısal class ID değerlerine bakarak elle birleştirmeyin.
* Hedef obje içeren görüntüler için boş label oluşturmayın.
* Test sonuçlarını kullanarak modeli tekrar tekrar ayarlamayın. Model ayarlamaları için validation sonuçlarını kullanın ve test verisini son değerlendirme için ayırın.
* Aynı videodan veya fotokapan olayından gelen komşu kareleri farklı split’lere dağıtmayın.
* Eğitimden önce ve son ZIP arşivini oluşturmadan önce doğrulama işlemini çalıştırın.

## Platform Notları

Programın klasör seçme, dosya yolu oluşturma, dataset işleme ve görüntü açma davranışları hem Windows hem de Linux dosya yolu yapılarını dikkate alır.

Linux’ta bir görüntüyü harici olarak açmak için `xdg-utils` paketinin sağladığı `xdg-open` gerekir.

Windows’ta varsayılan sistem görüntüleyicisi kullanılır.
</details>
