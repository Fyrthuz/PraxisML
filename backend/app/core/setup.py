import os

# Asegurar que los directorios locales de subida de archivos existen
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
TENANTS_DIR = os.path.join(DATA_DIR, "tenants")

def ensure_data_directories():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(TENANTS_DIR, exist_ok=True)

if __name__ == "__main__":
    ensure_data_directories()
    print("Directorios de almacenamiento local creados existosamente.")
