import bpy
import bmesh
import numpy as np
import sys

def create_pixel_perfect_pointcloud(obj, resolution=1024, output_path="C:/temp/pointcloud.ply"):
    """
    Generates a point cloud where points are arranged in a UV grid 
    and snapped to the 3D surface.
    One point per 'pixel' of the defined resolution.
    """
    
    print(f"Generating {resolution}x{resolution} ({resolution**2}) points...")
    
    # Create the Source Grid (The "Pixels")
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=resolution-1, 
        y_subdivisions=resolution-1, 
        size=1, 
        calc_uvs=True
    )
    grid = bpy.context.active_object
    
    grid.location = (0.5, 0.5, 0)
    bpy.ops.object.transform_apply(location=True)
    
    # Geometry Nodes Setup
    mod = grid.modifiers.new(name="GeoNodes", type='NODES')
    node_group = bpy.data.node_groups.new('SnapToUV', 'GeometryNodeTree')
    mod.node_group = node_group
    
    inputs = node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    outputs = node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    links = node_group.links
    nodes = node_group.nodes
    
    node_in = nodes.new('NodeGroupInput')
    node_out = nodes.new('NodeGroupOutput')
    node_obj_info = nodes.new('GeometryNodeObjectInfo')
    node_obj_info.inputs[0].default_value = obj  # Set the target object
    node_obj_info.transform_space = 'RELATIVE'
    
    node_pos = nodes.new('GeometryNodeInputPosition')
    node_sample = nodes.new('GeometryNodeSampleUVSurface')
    
    # We must tell the node to sample a Vector (XYZ), not a Float.
    node_sample.data_type = 'FLOAT_VECTOR'
    
    node_set_pos = nodes.new('GeometryNodeSetPosition')
    
    # Target Mesh -> Sample UV Surface Source
    links.new(node_obj_info.outputs['Geometry'], node_sample.inputs['Mesh'])
    
    # Grid Position -> Sample UV Surface UV (Use X and Y of grid position as U and V)
    links.new(node_pos.outputs['Position'], node_sample.inputs['Sample UV'])
    
    # Sampled Position -> Set Position
    links.new(node_in.outputs['Geometry'], node_set_pos.inputs['Geometry'])
    links.new(node_sample.outputs['Value'], node_set_pos.inputs['Position']) # <--- CORRECTED
    
    # Valid Selection (Only keep points that actually hit the mesh)
    node_delete = nodes.new('GeometryNodeDeleteGeometry')
    node_not = nodes.new('FunctionNodeBooleanMath')
    node_not.operation = 'NOT'
    
    links.new(node_set_pos.outputs['Geometry'], node_delete.inputs['Geometry'])
    links.new(node_sample.outputs['Is Valid'], node_not.inputs[0])
    links.new(node_not.outputs[0], node_delete.inputs['Selection'])
    
    links.new(node_delete.outputs['Geometry'], node_out.inputs['Geometry'])
    
    # Apply Modifier to Bake Geometry
    print("Applying geometry nodes...")
    bpy.context.view_layer.objects.active = grid
    bpy.ops.object.modifier_apply(modifier="GeoNodes")
    
    # Export to PLY
    print(f"Exporting to {output_path}...")
    
    bpy.ops.object.select_all(action='DESELECT') # Deselect everything
    grid.select_set(True)                       # Select just the grid
    bpy.context.view_layer.objects.active = grid # Ensure it's active

    bpy.ops.wm.ply_export(
        filepath=output_path, 
        export_selected_objects=True,
        export_colors=False # Set to True if you also bake colors
    )
    
    # Cleanup
    bpy.ops.object.select_all(action='DESELECT') 
    bpy.data.objects.remove(grid, do_unlink=True)
    print("Done.")

# Run it
if __name__ == "__main__":
    
    # 1. DEFINE YOUR OBJECT NAME HERE
    TARGET_OBJECT_NAME = "Default" # <--- Or "Suzanne", "MyModel", etc.
    
    # 2. Get object by name
    obj = bpy.data.objects.get(TARGET_OBJECT_NAME)
    
    # 3. Robust error checking
    if obj is None:
        print(f"ERROR: Object '{TARGET_OBJECT_NAME}' not found in the .blend file!")
        sys.exit(1) # Exit with an error code
    elif obj.type != 'MESH':
        print(f"ERROR: Object '{TARGET_OBJECT_NAME}' is not a MESH! (Found {obj.type})")
        sys.exit(1) # Exit with an error code
    
    print(f"Target object set to: {obj.name}")

    create_pixel_perfect_pointcloud(obj, resolution=1024)
