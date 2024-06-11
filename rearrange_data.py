import os
import sqlite3
import shutil

dataset_dir = 'dataset'
source_dir = 'dataset-clusters'

for filename in os.listdir(dataset_dir):
    file_path = os.path.join(dataset_dir, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

for cluster_dir in os.listdir(source_dir):
    cluster_path = os.path.join(source_dir, cluster_dir)
    if os.path.isdir(cluster_path):
        for filename in os.listdir(cluster_path):
            file_path = os.path.join(cluster_path, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, dataset_dir)

shutil.rmtree(source_dir)

conn = sqlite3.connect('customer_faces_data.db')
c = conn.cursor()
c.execute("SELECT id, image_path FROM customers")
rows = c.fetchall()

for row in rows:
    image_path = row[1]
    
    if not os.path.isfile(image_path):
        c.execute("DELETE FROM customers WHERE id=?", (row[0],))
        conn.commit()
        print(f"Deleted record with id {row[0]} because the associated picture '{image_path}' does not exist.")

conn.close()
