import os
import trimesh
from stl import mesh


def compute_convex_hull(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".STL"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.obj")
            try:
                STL_mesh = mesh.Mesh.from_file(input_path)
                vertices = STL_mesh.vectors.reshape(-1, 3)
                trimesh_mesh = trimesh.Trimesh(vertices=vertices, process=True)
                convex_hull = trimesh_mesh.convex_hull
                convex_hull.export(output_path, file_type='obj')
                print(f"Convex hull saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    input_directory = "meshes"
    output_directory = "meshes_convex"
    compute_convex_hull(input_directory, output_directory)
