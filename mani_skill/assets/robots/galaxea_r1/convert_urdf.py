import os
import xml.etree.ElementTree as ET

def convert_urdf(urdf_file, output_urdf_file, output_collision_dir):
    """
    Update the URDF file to replace collision mesh files with GLB while keeping visual meshes as STL.
    """
    # Parse the URDF file
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Ensure the output directory exists
    os.makedirs(output_collision_dir, exist_ok=True)

    for link in root.findall(".//link"):
        # Process the collision meshes
        collision = link.find("collision")
        if collision is not None:
            geometry = collision.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    if filename and filename.endswith(".STL"):
                        # Update collision mesh to use .glb
                        new_filename = os.path.join(output_collision_dir, os.path.basename(filename).replace(".STL", ".glb"))
                        mesh.set("filename", new_filename)

    # Save the updated URDF file
    tree.write(output_urdf_file, encoding="utf-8", xml_declaration=True)
    print(f"Updated URDF saved to: {output_urdf_file}")

# Example usage
urdf_file = "r1_upperbody.urdf"  # Path to the original URDF file
output_urdf_file = "r1_upperbody_glb.urdf"  # Path to save the updated URDF
output_collision_dir = "meshes_glb"  # Directory for collision GLB files

convert_urdf(urdf_file, output_urdf_file, output_collision_dir)
