import sqlite3
conn = sqlite3.connect('database/species_database.sqlite')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', tables)

if tables:
    cursor.execute("PRAGMA table_info(species)")
    columns = cursor.fetchall()
    print('\nSpecies columns:', [col[1] for col in columns])

    cursor.execute("SELECT * FROM species LIMIT 3")
    rows = cursor.fetchall()
    print('\nSample data:')
    for row in rows:
        print(row)

conn.close()
