# config/constants.py
import pandas as pd
import numpy as np

# =========================================================
# STAGE 1 CONSTANTS
# =========================================================
HOUSE_SUBTYPES = {
    "residence", "villa", "mixed building", "master house",
    "cottage", "bungalow", "chalet", "mansion"
}

APARTMENT_SUBTYPES = {
    "apartment", "ground floor", "penthouse", "duplex",
    "studio", "loft", "triplex", "student flat", "student housing"
}

YES_NO_COLS = [
    "leased", "running_water", "access_disabled", "preemption_right",
    "has_swimming_pool", "sewer_connection", "attic", "cellar",
    "entry_phone", "solar_panels", "planning_permission_granted",
    "alarm", "heat_pump", "surroundings_protected", "air_conditioning",
    "rain_water_tank", "security_door", "low_energy", "water_softener",
    "opportunity_for_professional"
]

NUMERIC_STR_COLS = ["frontage_width", "terrain_width_roadside"]

CORE_DTYPES = {
    "property_id": "string",
    "url": "string",
}

EXPECTED_MIN_COLUMNS = [
    "url",
    "property_id",
    "price",
]

# =========================================================
# STAGE 2.5 CONSTANTS (Geo Enrichment)
# =========================================================

PROVINCE_TO_REGION = {
    # --- Flanders ---
    'Provincie Antwerpen': 'Flanders',
    'Province d’Anvers': 'Flanders',
    'Province d\'Anvers': 'Flanders',

    'Provincie Oost-Vlaanderen': 'Flanders',
    'Province de Flandre orientale': 'Flanders',

    'Provincie West-Vlaanderen': 'Flanders',
    'Province de Flandre occidentale': 'Flanders',

    'Provincie Vlaams-Brabant': 'Flanders',
    'Province du Brabant flamand': 'Flanders',

    'Provincie Limburg': 'Flanders',
    'Province du Limbourg': 'Flanders',

    # --- Wallonia ---
    'Provincie Henegouwen': 'Wallonia',
    'Province du Hainaut': 'Wallonia',

    'Provincie Luik': 'Wallonia',
    'Province de Liège': 'Wallonia',

    'Provincie Namen': 'Wallonia',
    'Province de Namur': 'Wallonia',

    'Provincie Waals-Brabant': 'Wallonia',
    'Province du Brabant wallon': 'Wallonia',

    'Provincie Luxemburg': 'Wallonia',
    'Province du Luxembourg': 'Wallonia',

    # --- Brussels ---
    'Brussels Hoofdstedelijk Gewest': 'Brussels',
    'Région de Bruxelles-Capitale': 'Brussels'
}

PRICE_TABLE_PROVINCES = pd.DataFrame({
    "province": [
        'Provincie Antwerpen', 'Provincie Oost-Vlaanderen',
        'Provincie Henegouwen', 'Brussels Hoofdstedelijk Gewest',
        'Provincie Luik', 'Provincie West-Vlaanderen', 'Provincie Namen',
        'Provincie Waals-Brabant', 'Provincie Vlaams-Brabant',
        'Provincie Limburg', 'Provincie Luxemburg'
    ],
    "apt_avg_m2":  [2849, 2890, 1847, 3423, 2292, 3760, 2553, 3244, 3260, 2562, 2448],
    "house_avg_m2":[2419, 2257, 1411, 3308, 1708, 2048, 1673, 2342, 2539, 1926, 1620]
})

PRICE_TABLE_REGIONS = pd.DataFrame({
    "region":       ['Flanders', 'Wallonia', 'Brussels'],
    "apt_avg_m2":   [3133,      2387,       3423],
    "house_avg_m2": [2266,      1642,       3308]
})

PRICE_TABLE_BELGIUM = {
    "apartment": {"avg": 3091, "min": 2188, "max": 4358},
    "house":     {"avg": 2076, "min": 1223, "max": 3296}
}
