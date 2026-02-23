# src/core/data_loader.py
import os
import csv

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _normalize_team(team: str) -> str:
    if team is None:
        return ""
    t = str(team).strip()
    # unify grpX -> gpX (important!)
    if t.startswith("grp"):
        t = "gp" + t[3:]
    return t

def _resolve_image_root(images_root: str, team_filter: str | None) -> str:
    """
    If team_filter is provided (e.g., 'gp5' or 'grp5'),
    only scan images_root/<team>/.
    """
    if not team_filter:
        return images_root

    t = _normalize_team(team_filter)
    team_dir = os.path.join(images_root, t)
    if os.path.isdir(team_dir):
        return team_dir

    # fallback: maybe user passes exact folder name
    team_dir2 = os.path.join(images_root, str(team_filter).strip())
    if os.path.isdir(team_dir2):
        return team_dir2

    # if folder not found, keep original root (so pipeline still runs)
    return images_root

def iter_image_files(root_dir, team_filter=None):
    """
    Yield absolute image paths under root_dir recursively.
    If team_filter is set, only yield images under root_dir/<team_filter>/.
    Example:
        iter_image_files(".../data/images", team_filter="gp5")
        -> scans only .../data/images/gp5/
    """
    scan_root = _resolve_image_root(root_dir, team_filter)

    for dirpath, _, filenames in os.walk(scan_root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTS:
                yield os.path.join(dirpath, fn)

def _to_float_fr_or_none(x):
    """Parse French decimals; return None for NAN/empty."""
    if x is None:
        return None
    s = str(x).strip().strip('"').strip()
    if s == "":
        return None
    s_up = s.upper()
    if s_up in ("NAN", "NA", "NONE"):
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

def load_annotations(csv_path, team_filter=None):
    """
    Load annotations from CSV.
    If team_filter is provided (e.g., 'gp5' or 'grp5'),
    keep only rows belonging to that team.
    """
    team_filter_norm = _normalize_team(team_filter) if team_filter else ""

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

        # >>> filter by team if requested
        if team_filter_norm and team != team_filter_norm:
            continue

        cnt = _to_int_safe(r[c_cnt])
        euros = _to_float_fr_or_none(r[c_val])

        ann[(team, img_name)] = {"count": cnt, "euros": euros, "team_raw": team_raw}

    return ann