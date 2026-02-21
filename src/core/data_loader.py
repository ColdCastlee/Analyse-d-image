# src/core/data_loader.py
import os
import csv

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_image_files(root_dir):
    """Yield absolute image paths under root_dir recursively."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTS:
                yield os.path.join(dirpath, fn)

def _normalize_team(team: str) -> str:
    if team is None:
        return ""
    t = team.strip()
    # unify grpX -> gpX (important!)
    if t.startswith("grp"):
        t = "gp" + t[3:]
    return t

def _to_float_fr_or_none(x):
    """Parse French decimals; return None for NAN/empty."""
    if x is None:
        return None
    s = str(x).strip().strip('"').strip()
    if s == "":
        return None
    s_up = s.upper()
    if s_up == "NAN" or s_up == "NA" or s_up == "NONE":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def basename_only(path):
    return os.path.basename(path)

def _to_float_fr(x):
    """Parse '6,18' or '6.18' -> float"""
    if x is None:
        return None
    s = str(x).strip().strip('"').strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def _to_int_safe(x):
    if x is None:
        return None
    s = str(x).strip().strip('"').strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None

def load_annotations(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)

    header_idx = None
    for i, r in enumerate(rows):
        if len(r) >= 2 and any("Nom image" in c for c in r):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Cannot find header row containing 'Nom image' in annotations.csv")

    header = [c.strip() for c in rows[header_idx]]

    def col(name):
        for j, c in enumerate(header):
            if c.strip() == name:
                return j
        return None

    c_img  = col("Nom image")
    c_cnt  = col("Nombre de pièces")
    c_val  = col("Valeur monétaire €")
    c_team = col("Identifiant équipe")

    if None in (c_img, c_cnt, c_val, c_team):
        raise RuntimeError(f"Missing required columns in annotations header: {header}")

    ann = {}
    for r in rows[header_idx + 1:]:
        if len(r) < max(c_img, c_cnt, c_val, c_team) + 1:
            continue

        img_name = r[c_img].strip()
        if img_name == "":
            continue

        team_raw = r[c_team].strip()
        team = _normalize_team(team_raw)

        cnt = _to_int_safe(r[c_cnt])
        euros = _to_float_fr_or_none(r[c_val])

        ann[(team, img_name)] = {"count": cnt, "euros": euros, "team_raw": team_raw}

    return ann
