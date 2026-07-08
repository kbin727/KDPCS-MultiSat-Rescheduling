"""Crop PDF whitespace for all generated paper figures."""
import os
import fitz

FIG_DIR = r"E:\论文材料\DynamicReschedule_MultiSat\results\KDPCS_overleaf_clean_20260709_polished_work\figures"

pdfs_to_crop = []
# top-level PDFs
for f in os.listdir(FIG_DIR):
    if f.endswith(".pdf"):
        pdfs_to_crop.append(os.path.join(FIG_DIR, f))
# pareto subfolder
pareto_dir = os.path.join(FIG_DIR, "pareto_final_all18_full_20260705")
if os.path.isdir(pareto_dir):
    for f in os.listdir(pareto_dir):
        if f.endswith(".pdf"):
            pdfs_to_crop.append(os.path.join(pareto_dir, f))

pdfs_to_crop.sort()
print(f"Found {len(pdfs_to_crop)} PDFs to crop.\n")

print(f"{'File':<55} {'size_pt':>10} {'drawings':>9} {'texts':>6} {'raster':>7}")
print("-" * 92)

for path in pdfs_to_crop:
    name = os.path.basename(path)
    doc = fitz.open(path)
    page = doc[0]
    rect = page.rect

    # Check content types
    has_vector = len(page.get_drawings()) > 0
    has_image = len(page.get_images(full=True)) > 0
    n_drawings = len(page.get_drawings())
    n_texts = len(page.get_text("blocks"))

    # Compute content bbox
    bbox = None
    for d in page.get_drawings():
        r = d["rect"]
        if bbox is None:
            bbox = fitz.Rect(r)
        else:
            bbox = bbox | r
    for tb in page.get_text("blocks"):
        r = fitz.Rect(tb[:4])
        if bbox is None:
            bbox = fitz.Rect(r)
        else:
            bbox = bbox | r
    for img_info in page.get_image_info():
        r = fitz.Rect(img_info["bbox"])
        if bbox is None:
            bbox = fitz.Rect(r)
        else:
            bbox = bbox | r
    if bbox is None:
        bbox = rect

    # Clamp and pad
    pad = 1.0
    new_x0 = max(rect.x0, bbox.x0 - pad)
    new_y0 = max(rect.y0, bbox.y0 - pad)
    new_x1 = min(rect.x1, bbox.x1 + pad)
    new_y1 = min(rect.y1, bbox.y1 + pad)
    new_bbox = fitz.Rect(new_x0, new_y0, new_x1, new_y1)

    if new_bbox.is_valid and not new_bbox.is_empty and \
            (new_bbox.width < rect.width or new_bbox.height < rect.height):
        try:
            page.set_cropbox(new_bbox)
        except Exception:
            pass

    tmp = path + ".tmp"
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp, path)

    # Verify
    doc2 = fitz.open(path)
    p2 = doc2[0]
    r2 = p2.rect
    doc2.close()

    raster = "yes" if has_image else "no"
    print(f"{name:<55} {r2.width:.0f}x{r2.height:.0f}  {n_drawings:>9} {n_texts:>6} {raster:>7}")

print("\nAll PDFs cropped and verified.")
