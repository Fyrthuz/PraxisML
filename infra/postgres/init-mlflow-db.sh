#!/bin/bash
set -e

# Crea la bbdd de mlflow además de la de praxisml_db que se crea por defecto.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
	CREATE USER mlflow WITH PASSWORD 'mlflow';
	CREATE DATABASE mlflow_db;
	GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;
	\connect mlflow_db
	GRANT ALL ON SCHEMA public TO mlflow;
EOSQL
