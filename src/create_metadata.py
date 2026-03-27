import os
import json
import glob

def build_metadata(base_path):
    metadata = []

    for label_name, label in [
        ("real", 0),
        ("fake_fvra", 1),
        ("fake_fvfa", 1)
    ]:
        folder = os.path.join(base_path, label_name)

        files = glob.glob(folder + "/*.frames.npy")

        for f in files:
            prefix = f.replace(".frames.npy", "")
            video_id = os.path.basename(prefix)

            metadata.append({
                "video_id": video_id,
                "label": label,
                "type": label_name,
                "frames_path": prefix + ".frames.npy",
                "height_path": prefix + ".height.txt",
                "width_path": prefix + ".width.txt",
                "num_frames_path": prefix + ".num_frames.txt"
            })

    return metadata


metadata = build_metadata("C:\\Users\\Guilherme Monteiro\\Desktop\\TCC\\data\\extracted")

with open("C:\\Users\\Guilherme Monteiro\\Desktop\\TCC\\data\\extracted\\metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)