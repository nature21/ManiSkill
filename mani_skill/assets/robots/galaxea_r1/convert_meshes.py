import os
import xml.etree.ElementTree as ET
import trimesh
import coacd


def parse_urdf(urdf_file):
    """Parse a URDF file to find all STL mesh paths."""
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    stl_files = []

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and filename.endswith(".STL"):
            stl_files.append(filename)

    # remove duplicates
    stl_files = list(set(stl_files))

    return stl_files


def convert_stl_to_glb(stl_files, output_dir):
    """Convert STL files to GLB files with convex decomposition."""
    os.makedirs(output_dir, exist_ok=True)

    for stl_file in stl_files:
        print(f"Processing {stl_file}...")

        # Load the STL file
        mesh = trimesh.load(stl_file, force="mesh")

        # Run COACD to decompose into convex parts
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        result = coacd.run_coacd(coacd_mesh)  # List of convex hulls
        mesh_parts = []
        for vs, fs in result:
            mesh_parts.append(trimesh.Trimesh(vs, fs))

        scene = trimesh.Scene()
        for p in mesh_parts:
            scene.add_geometry(p)
        # Save the GLB file
        glb_filename = os.path.join(output_dir, os.path.basename(stl_file).replace(".STL", ".glb"))
        scene.export(glb_filename)
        print(f"Saved: {glb_filename}")

    print("All STL files processed.")


def convert_urdf_stl_to_glb(urdf_file, output_dir):
    """Main function to convert URDF's STL meshes to GLB meshes."""
    stl_files = parse_urdf(urdf_file)
    if not stl_files:
        print("No STL files found in the URDF.")
        return

    print(f"Found {len(stl_files)} STL files in the URDF.")
    convert_stl_to_glb(stl_files, output_dir)


# Example usage
urdf_file = "r1.urdf"  # Path to your URDF file
output_dir = "meshes_glb"  # Directory to save GLB files
convert_urdf_stl_to_glb(urdf_file, output_dir)
