import trimesh

# Load the STL file
mesh = trimesh.load_mesh("/home/yuyao/abc/ManiSkill/mani_skill/assets/robots/g1_humanoid/meshes/left_ankle_pitch_link.STL")

# Check if the mesh is already convex decomposed
if isinstance(mesh, trimesh.Scene):
    # If it's a scene, count the number of meshes (convex parts)
    print(f"Number of convex parts: {len(mesh.geometry)}")
else:
    # Single mesh: check for convexity
    is_convex = mesh.is_convex
    print(f"Is the mesh convex? {is_convex}")

    # Decompose into convex parts (if needed)
    if not is_convex:
        convex_parts = mesh.convex_decomposition()
        print(f"Number of convex parts after decomposition: {len(convex_parts)}")
