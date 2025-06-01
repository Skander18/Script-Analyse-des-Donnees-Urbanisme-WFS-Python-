import geopandas as gpd
import pandas as pd
import requests
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Configuration
WFS_URL = "https://data.geopf.fr/wfs/ows"
LAYER_NAME = "wfs_du:zone_urba"
CRS = "EPSG:4326"
DEPT_CODES = ['13', '69', '75']
MAX_FEATURES = 500

# Emprises optimisées
BBOX_TARGET = {
    '13': (4.7, 43.2, 5.4, 43.5),  # Marseille centre
    '69': (4.7, 45.7, 4.9, 45.8),  # Lyon centre
    '75': (2.3, 48.8, 2.4, 48.9)   # Paris centre
}
TILE_SIZE = 0.1  # Environ 11 km

def get_features(bbox, dept_code):
    """Récupère les données avec gestion d'erreur améliorée"""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": LAYER_NAME,
        "srsName": CRS,
        "outputFormat": "application/json",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{CRS}",
        "count": MAX_FEATURES
    }
    
    try:
        response = requests.get(WFS_URL, params=params, timeout=30)
        response.raise_for_status()
        return gpd.read_file(BytesIO(response.content))
    except Exception as e:
        print(f"Erreur {dept_code} {bbox}: {str(e)[:100]}...")
        return None

def clean_data(gdf, dept_code):
    """Nettoyage des données"""
    if gdf is None or gdf.empty:
        return None
    
    # Conversion des types de données problématiques
    for col in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str)
    
    # Extraction du code département (méthode plus robuste)
    gdf['dep'] = gdf['partition'].str.extract(r'(\d{2})')  # Capture les 2 premiers chiffres
    
    # Filtrage et vérification
    filtered = gdf[gdf['dep'] == dept_code].copy()
    
    if filtered.empty:
        print(f"Aucune donnée valide pour {dept_code} dans cette tuile")
        return None
    
    # Calcul superficie
    filtered['area_km2'] = filtered.geometry.to_crs(epsg=3035).area / 1e6
    return filtered[['gpu_doc_id', 'partition', 'dep', 'area_km2', 'geometry']]

def process_department(dept_code):
    """Traitement par département avec journalisation"""
    print(f"\n=== Traitement département {dept_code} ===")
    bbox = BBOX_TARGET[dept_code]
    tiles = []
    
    # Génération des tuiles
    x = bbox[0]
    while x < bbox[2]:
        y = bbox[1]
        while y < bbox[3]:
            tiles.append((x, y, min(x + TILE_SIZE, bbox[2]), min(y + TILE_SIZE, bbox[3])))
            y += TILE_SIZE
        x += TILE_SIZE
    
    # Traitement des tuiles
    results = []
    for i, tile in enumerate(tiles, 1):
        print(f"Tuile {i}/{len(tiles)} - {tile}")
        data = get_features(tile, dept_code)
        cleaned = clean_data(data, dept_code)
        if cleaned is not None:
            results.append(cleaned)
        time.sleep(0.5)
    
    return pd.concat(results, ignore_index=True) if results else None

def generate_stats(gdf):
    """Génère les statistiques avec vérification"""
    if gdf.empty:
        return None
    
    stats = gdf.groupby('dep').agg(
        nb_documents=('gpu_doc_id', 'count'),
        total_area=('area_km2', 'sum')
    ).reset_index()
    stats['density'] = stats['nb_documents'] / stats['total_area']
    return stats

def save_results(gdf, stats):
    """Sauvegarde sécurisée des résultats"""
    # Vérification des types avant export
    for col in gdf.select_dtypes(include=['datetime', 'timedelta']):
        gdf[col] = gdf[col].astype(str)
    
    # Export GPKG
    try:
        gdf.to_file("documents_urbanisme.gpkg", driver="GPKG")
        print(" Fichier GPKG créé")
    except Exception as e:
        print(f" Erreur GPKG: {str(e)}")
        # Fallback en GeoJSON
        gdf.to_file("documents_urbanisme.geojson", driver="GeoJSON")
        print(" Fichier GeoJSON créé (fallback)")
    
    # Export CSV
    stats.to_csv("statistiques_departements.csv", index=False)
    print(" Fichier CSV créé")

def main():
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_department, dept): dept for dept in DEPT_CODES}
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)
    
    if not results:
        print("\n Aucune donnée valide n'a pu être récupérée")
        return
    
    final_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=CRS)
    stats = generate_stats(final_gdf)
    
    if stats is not None:
        print("\n Résultats finaux:")
        print(stats)
        save_results(final_gdf, stats)
    
    print(f"\n Temps total: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()
