# -*- coding: utf-8 -*-
"""
creat.py
========
Single-coordinate-system 3-channel image generation script.

Differences from the old 1.py / separate-3-channels scripts:
1. Reads only ONE coordinate file orthogonality.csv (label, x, y) instead of the
   three background maps 1_2 / 1_3 / 2_3.
2. Image size is 384x196 (width x height): each pathway's (x, y) coordinate is
   linearly scaled onto the 384x196 grid.
3. The three channels are three independent matrices for [drug1 / drug2 / cell line]
   (each Z-score normalized then stacked), instead of the old "three coordinate
   backgrounds overlaid" channels.
   Output tensor shape: (num_samples, 3, 384, 196)   # spatial dim 0 = X = 384, dim 1 = Y = 196

Dataset selection (--dataset argument):
  ONEIL       → under data/ONEIL/: drug_pathway.csv, cell-line.csv, ONEIL_socre.csv, State_predict.csv
  NCI-ALMANAC → under data/NCI-ALMANAC/: drugs_pathways.csv, cell_line.csv, NCI-ALMANAC_socre.csv
                (no State_predict.csv; when the activity file is missing, fall back
                 to binary counting as "no activity mapping")
"""
import os
import csv
import argparse
import numpy as np
import pandas as pd

# ---- Dataset selection (command-line argument) ----
_parser = argparse.ArgumentParser(description="Generate 3-channel drug synergy images.")
_parser.add_argument('--dataset', choices=['ONEIL', 'NCI-ALMANAC'], default='ONEIL',
                     help="dataset to use: ONEIL or NCI-ALMANAC (default: ONEIL)")
_args = _parser.parse_args()
DATASET = _args.dataset

# ---- Path configuration (auto-located relative to this script) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../PPSynergy/data
DATA_DIR = os.path.join(BASE_DIR, DATASET)                      # .../PPSynergy/data/<dataset name>

ORTHO_FILE         = os.path.join(BASE_DIR, 'orthogonality.csv')   # single coordinate system (label,x,y)
DRUG_PATHWAY_FILE  = os.path.join(DATA_DIR, 'drug_pathway.csv' if DATASET == 'ONEIL' else 'drugs_pathways.csv')  # drug-pathway data
CELL_PATH_FILE     = os.path.join(DATA_DIR, 'cell-line.csv' if DATASET == 'ONEIL' else 'cell_line.csv')         # cell line-pathway expression
SYNERGY_FILE       = os.path.join(DATA_DIR, 'ONEIL_socre.csv' if DATASET == 'ONEIL' else 'NCI-ALMANAC_socre.csv')  # sample data
PATHWAYS_FILE      = os.path.join(DATA_DIR, 'pathways.csv')        # pathway list
ACTIVITY_FILE      = os.path.join(DATA_DIR, 'State_predict.csv')   

GRID_W = 384      # image width (pixels along X)
GRID_H = 196     # image height (pixels along Y)


def read_csvs(filepath):
    """Read a CSV into a dict: {name: [element lists, ...]}"""
    result = {}
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            name = row[0]
            if name not in result:
                result[name] = []
            result[name].append(row[1:])
    return result


def build_result_list(filepath, grid_w, grid_h):
    """Read the coordinate file (label,x,y) and linearly scale (x, y) onto the grid_w x grid_h grid (width x height).

    Returns:
        coord_map: {label: (row, col), ...} (dict, O(1) lookup)
        (x_min, x_max, y_min, y_max): used for consistent coordinate mapping later
    """
    data = pd.read_csv(filepath)
    x_min, x_max = float(data['x'].min()), float(data['x'].max())
    y_min, y_max = float(data['y'].min()), float(data['y'].max())

    def to_coord(x, y):
        col = int(round((x - x_min) / (x_max - x_min) * (grid_w - 1)))  # X -> first dim (0..383)
        row = int(round((y - y_min) / (y_max - y_min) * (grid_h - 1)))  # Y -> second dim (0..195)
        return col, row

    coord_map = {}
    for label, x, y in zip(data['label'], data['x'], data['y']):
        coord_map[label] = to_coord(x, y)
    return coord_map, (x_min, x_max, y_min, y_max)


def find_element(coord_map, target):
    """Look up the grid coordinate of a pathway name in the coordinate map (O(1))"""
    return coord_map.get(target)


def load_activity_data(filepath):
    """Read State_predict.csv (cell_name, ps_drug + pathway activity columns) into a (cell, drug) -> activity dict.

    If the file is missing or empty, return {} (i.e. "no activity mapping"), so the
    drug matrix falls back to the drug_pathway binary counting logic (strategy 2 of get_matrix_drug).
    """
    if not os.path.exists(filepath):
        print(f"Warning: activity file not found {filepath}, falling back to no-activity mapping (drug_pathway binary counting)")
        return {}
    df = pd.read_csv(filepath)
    if df.empty:
        print("Warning: activity file has no data rows (header only), all drug matrices will use the drug_pathway counting fallback")
        return {}
    activity_dict = df.set_index(['cell_name', 'ps_drug']).to_dict(orient='index')
    return activity_dict


def get_matrix_drug(drug_name, cell_name, result_dict, activity_dict, coord_map, grid_h, grid_w):
    """Single drug matrix (grid_w, grid_h) = (X, Y) = (384, 196).

    Strategy 1: if (cell, drug) exists in the activity file, fill each pathway
                activity value into its corresponding coordinate;
    Strategy 2: otherwise fall back to drug_pathway.csv, counting (+1) at the
                coordinates of the pathways involved by this drug.
    """
    empty_matrix = np.zeros((grid_w, grid_h), dtype=float)

    # Strategy 1: pathway activity from the activity file
    if activity_dict and (cell_name, drug_name) in activity_dict:
        row_data = activity_dict[(cell_name, drug_name)]
        for p_name, val in row_data.items():
            coord = find_element(coord_map, p_name)
            if coord:
                empty_matrix[coord] = val
        return empty_matrix

    # Strategy 2: drug_pathway counting fallback
    value = result_dict.get(drug_name)
    if value:
        for item in value:
            for target in item:
                if target != '':
                    coord = find_element(coord_map, target)
                    if coord:
                        empty_matrix[coord] += 1
    return empty_matrix


def get_matrix_cell(value, pathway_names, coord_map, grid_h, grid_w):
    """Single cell-line matrix (grid_w, grid_h) = (X, Y) = (384, 196): fill each pathway's expression value into its coordinate"""
    empty_matrix = np.zeros((grid_w, grid_h), dtype=float)

    coords = []
    for target in pathway_names:
        if target != '':
            coords.append(find_element(coord_map, target))

    value_data = []
    if value:
        for sublist in value:
            for item in sublist:
                value_data.append(float(item))

    # Fill only when the number of pathways matches the number of expression values (cell-path row aligned with the pathways list)
    if len(coords) == len(value_data):
        for coord, val in zip(coords, value_data):
            if coord:
                empty_matrix[coord] = val
    else:
        print(f"Warning: #pathways ({len(coords)}) != #expression values ({len(value_data)}), skipping cell-line matrix fill")

    return empty_matrix


def normalize(m):
    """Z-score normalize (only center if the standard deviation is 0)"""
    s = np.std(m)
    return (m - np.mean(m)) / s if s != 0 else m - np.mean(m)


# ---- Load data ----
result = read_csvs(DRUG_PATHWAY_FILE)
cell_line = read_csvs(CELL_PATH_FILE)
drug_synergy = pd.read_csv(SYNERGY_FILE)
pathways_df = pd.read_csv(PATHWAYS_FILE)
pathway_names = pathways_df['pathways'].tolist()
activity_lookup = load_activity_data(ACTIVITY_FILE)
coord_map, coord_range = build_result_list(ORTHO_FILE, GRID_W, GRID_H)


def get_drug_matrix_v3():
    matrix_drug_cell = []

    drug1_arr = np.array(drug_synergy['drug1'])
    drug2_arr = np.array(drug_synergy['drug2'])
    cell_arr = np.array(drug_synergy['cell'])

    for i in range(len(drug1_arr)):
        if i % 10 == 0:
            print(f"Processing: {i}/{len(drug1_arr)}")

        d1, d2, c = drug1_arr[i], drug2_arr[i], cell_arr[i]
        v3 = cell_line.get(c)

        # Three independent matrices: drug1 / drug2 / cell line
        m1 = get_matrix_drug(d1, c, result, activity_lookup, coord_map, GRID_H, GRID_W)
        m2 = get_matrix_drug(d2, c, result, activity_lookup, coord_map, GRID_H, GRID_W)
        m_cell = get_matrix_cell(v3, pathway_names, coord_map, GRID_H, GRID_W)

        # Z-score normalize each channel separately, then stack as (3, 384, 196)
        c1, c2, c3 = normalize(m1), normalize(m2), normalize(m_cell)
        matrix_drug_cell.append(np.stack((c1, c2, c3), axis=0))

    return np.array(matrix_drug_cell, dtype=np.float32)


# ---- Run and save ----
if __name__ == '__main__':
    final_matrix = get_drug_matrix_v3()
    print("Final Matrix Shape:", final_matrix.shape)  # expected (num_samples, 3, 384, 196)

    output_path = os.path.join(BASE_DIR, f'{DATASET}.npy')
    np.save(output_path, final_matrix)
    print(f"Saved successfully to: {output_path}")
