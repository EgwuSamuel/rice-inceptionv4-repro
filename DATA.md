# Data provenance

## Primary dataset: Kaggle `bahribahri/riceleaf`

- URL: https://www.kaggle.com/datasets/bahribahri/riceleaf (version 1, 23 June 2020; licence listed as "Unknown").
- 16,000 JPEG images, 4,000 per class, in the folders `Bacterial leaf blight`, `Brown spot`, `Health`, `Leaf smut`. Each file is about 0.7–2.3 kB.
- Every filename has the form `<Class>_original_<source photo>_<uuid>.<ext>`, which is the output pattern of the Augmentor library. Parsing the filenames (Kaggle API file listing, checked 14 Oct 2026) gives:

| Folder | Images | Distinct source photographs |
|---|---|---|
| Bacterial leaf blight | 4,000 | 40 |
| Brown spot | 4,000 | 40 |
| Health | 4,000 | 40 |
| Leaf smut | 4,000 | 40 |

- The 120 disease source photographs (`DSC_*.JPG`) match, file name for file name, the 120 images of the *Rice Leaf Diseases* dataset (Prajapati et al.; UCI Machine Learning Repository; Kaggle mirror `vbookshelf/rice-leaf-diseases`).
- The 40 healthy source photographs (`IMG_20190419_*.jpg`) come from a different, undocumented source. A classifier can therefore separate *Healthy* from the diseases using acquisition cues (camera, background, resolution) rather than leaf symptoms. The external test (E3) and the Grad-CAM maps (E4) are designed to expose this.

Because each source photograph appears about 100 times, `ricedx.data.build_index` assigns every image to its source group, and all evaluation splits keep each group entirely within one partition.

## External test set: Paddy Doctor

- Kaggle competition `paddy-disease-classification` (training images, 480×640 px, collected in Tamil Nadu, India). Only the classes shared with the primary dataset are used: `bacterial_leaf_blight` → Bacterial leaf blight, `brown_spot` → Brown spot, `normal` → Healthy. Leaf smut has no counterpart.
- Paddy Doctor is used **only for testing**; no model ever trains on it.

## Fallback for datasets without source filenames

If filenames do not encode the source, `build_index` groups near-duplicates with a 64-bit DCT perceptual hash (Hamming distance ≤ 6) and union-find.
