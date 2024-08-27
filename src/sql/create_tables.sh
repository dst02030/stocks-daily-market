#!/bin/bash
host=172.20.10.3
port=5432
username=postgres
database=$DB_NAME
echo $database

file_path="$(cd "$(dirname "$0")" && pwd)/create_sql"


echo "[START] Create schemas in $host:$port.."

schemas="$(ls "$file_path"/schema_*)" 
tables="$(ls "$file_path"/table_*)"
paths=("${schemas[@]}" "${tables[@]}")

# CREATE STARTS
for file_path in ${paths[@]};
do
    echo "#######################################################################################"
    echo "`date -Iseconds` [START] CREATE table: $file_path"
    
    echo "PGPASSWORD='******' psql -h $host -p $port -U $username"
    echo $(PGPASSWORD=$DB_PASSWORD psql -h $host -p $port -U $username -f $file_path -d $database)
done

