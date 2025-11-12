import csv

def getMetaData(csv_path="../midiDataTest/commu_meta.csv", track_id="commu00001"):
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next((r for r in reader if r.get("id") == track_id), None)

    if row is None:
        raise ValueError(f"Track id {track_id} not found")
    
    track_role = row["track_role"]

    return track_role

print(getMetaData(track_id="commu00003"))
