import os

path = os.path.join(os.getcwd(), "data_schema", "schema.yaml")
print("Path:", path)
print("Exists:", os.path.exists(path))
