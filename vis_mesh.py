import numpy as np
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("./experiments/14deg_submarine/meshes/00052316.ply")
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
