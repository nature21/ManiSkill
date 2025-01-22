import os
import tempfile
import trimesh
import coacd
import pybullet as p


def decompose_files(mesh_files, output_dir, method='coacd'):
    """Convert STL files to GLB files with convex decomposition."""
    os.makedirs(output_dir, exist_ok=True)
    for mesh_file in mesh_files:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f"Processing {mesh_file} with {method} ...")
        if method == 'vhacd':
            obj_filename = os.path.join(output_dir, '.'.join(os.path.basename(mesh_file).split('.')[:-1] + ['obj']))
            mesh = trimesh.load(mesh_file, force="mesh")
            mesh.export(obj_filename)
            with tempfile.NamedTemporaryFile(delete=True) as log_file:
                kwargs = {}
                p.vhacd(obj_filename, obj_filename, log_file.name, **kwargs)
        elif method == 'coacd':
            mesh = trimesh.load(mesh_file, force="mesh")
            coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            result = coacd.run_coacd(coacd_mesh)  # List of convex hulls
            mesh_parts = []
            for vs, fs in result:
                mesh_parts.append(trimesh.Trimesh(vs, fs))

            scene = trimesh.Scene()
            for part in mesh_parts:
                scene.add_geometry(part)
            # Save the GLB file
            glb_filename = os.path.join(output_dir, '.'.join(os.path.basename(mesh_file).split('.')[:-1] + ['glb']))
            scene.export(glb_filename)
            print(f"Saved: {glb_filename}")
        else:
            raise NotImplementedError(f"Method {method} not implemented.")
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    print("All mesh files processed.")


def decompose_all_meshes(input_dir, output_dir, method='coacd'):
    mesh_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                  f.endswith('.stl') or f.endswith('.STL')]
    decompose_files(mesh_files, output_dir, method)


if __name__ == '__main__':
    input_dir = 'meshes'
    method = 'coacd'
    # method = 'vhacd'
    output_dir = f'meshes_{method}'
    decompose_all_meshes(input_dir, output_dir, method)