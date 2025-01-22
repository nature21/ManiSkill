import os
import numpy as np
import trimesh
from stl import mesh

def process_and_compute_convex_hull(input_stl, output_obj):
    """
    Load an STL file, remove all points with x < -0.03, compute the convex hull, and save as an OBJ file.

    :param input_stl: Path to the input STL file.
    :param output_obj: Path to save the output OBJ file.
    """
    try:
        # Load the STL file
        stl_mesh = mesh.Mesh.from_file(input_stl)
        vertices = stl_mesh.vectors.reshape(-1, 3)

        # Filter points where x >= -0.03
        filtered_vertices = vertices[vertices[:, 0] >= -0.03]

        if len(filtered_vertices) == 0:
            print("No vertices remaining after filtering.")
            return

        # Create a Trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=filtered_vertices, process=True)

        # Compute the convex hull
        convex_hull = trimesh_mesh.convex_hull

        # Export the convex hull to OBJ format
        convex_hull.export(output_obj, file_type='obj')
        print(f"Convex hull saved: {output_obj}")
    except Exception as e:
        print(f"Error processing {input_stl}: {e}")

if __name__ == "__main__":
    filenames = ['left_gripper_link1.STL', 'left_gripper_link2.STL', 'right_gripper_link1.STL', 'right_gripper_link2.STL']
    input_dir = 'meshes'
    output_dir = 'meshes_convex'
    for filename in filenames:
        input_stl_path = os.path.join(input_dir, filename)
        output_obj_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.obj")
        process_and_compute_convex_hull(input_stl_path, output_obj_path)
