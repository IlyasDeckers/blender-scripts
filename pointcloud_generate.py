import bpy
import bmesh
import numpy as np
from mathutils import Vector
import random

# ============================================
# ADAPTIVE POINTCLOUD GENERATOR
# With automatic texture baking to vertex colors
# ============================================

def bake_texture_to_vertex_colors(obj):
    """
    Sample textures at VERTICES (not loops) for much faster baking
    Vertex colors will be interpolated across faces
    """
    print("=" * 50)
    print("BAKING TEXTURES TO VERTEX COLORS")
    print("=" * 50)
    
    mesh = obj.data
    
    # Check if object has UVs and materials
    if not mesh.uv_layers or not obj.material_slots:
        print("! Object has no UVs or materials, skipping bake")
        return False
    
    # Create vertex color layer
    if "BakedColor" in mesh.vertex_colors:
        vc_layer = mesh.vertex_colors["BakedColor"]
    else:
        vc_layer = mesh.vertex_colors.new(name="BakedColor")
    
    mesh.vertex_colors.active = vc_layer
    uv_layer = mesh.uv_layers.active.data
    
    # Build vertex to loops mapping (which loops reference each vertex)
    print(f"Building vertex->loop mapping for {len(mesh.vertices)} vertices...")
    vertex_to_loops = {v.index: [] for v in mesh.vertices}
    
    for poly in mesh.polygons:
        for loop_idx, vert_idx in zip(poly.loop_indices, poly.vertices):
            vertex_to_loops[vert_idx].append((loop_idx, poly.material_index))
    
    # Cache material info
    print("Preparing materials...")
    material_cache = {}
    
    for mat_idx, slot in enumerate(obj.material_slots):
        mat = slot.material
        if not mat:
            material_cache[mat_idx] = {'color': (0.8, 0.8, 0.8, 1.0), 'image': None}
            continue
        
        color = (0.8, 0.8, 0.8, 1.0)
        texture_image = None
        
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    texture_image = node.image
                    break
            
            if not texture_image:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        color = tuple(node.inputs['Base Color'].default_value)
                        break
        
        material_cache[mat_idx] = {'color': color, 'image': texture_image}
    
    # Sample color for each VERTEX (much fewer than loops)
    print(f"Sampling textures for {len(mesh.vertices)} vertices...")
    vertices_processed = 0
    
    for vert_idx, loops_info in vertex_to_loops.items():
        if not loops_info:
            continue
        
        # Use first occurrence of this vertex to sample color
        loop_idx, mat_idx = loops_info[0]
        mat_info = material_cache.get(mat_idx, {'color': (0.8, 0.8, 0.8, 1.0), 'image': None})
        
        if mat_info['image']:
            uv = uv_layer[loop_idx].uv
            image = mat_info['image']
            width, height = image.size
            
            x = int(max(0, min(0.9999, uv[0])) * width)
            y = int(max(0, min(0.9999, uv[1])) * height)
            pixel_idx = (y * width + x) * 4
            
            try:
                color = (
                    image.pixels[pixel_idx],
                    image.pixels[pixel_idx + 1],
                    image.pixels[pixel_idx + 2],
                    image.pixels[pixel_idx + 3]
                )
            except:
                color = mat_info['color']
        else:
            color = mat_info['color']
        
        # Apply this color to ALL loops that use this vertex
        for loop_idx, _ in loops_info:
            vc_layer.data[loop_idx].color = color
        
        vertices_processed += 1
        if vertices_processed % 50000 == 0:
            print(f"  Processed {vertices_processed}/{len(mesh.vertices)} vertices...")
    
    print("✓ Texture baking complete")
    return True


class AdaptivePointcloudGenerator:
    def __init__(self, obj, base_points=1000000, detail_multiplier=5.0, 
                 noise_strength=0.001, layers=3, use_vertex_colors=True):
        """
        obj: Blender mesh object
        base_points: Base number of points to distribute
        detail_multiplier: How many more points in high-curvature areas
        noise_strength: Amount of position randomization
        layers: Number of detail passes
        use_vertex_colors: Use baked vertex colors for accurate texture sampling
        """
        self.obj = obj
        self.base_points = base_points
        self.detail_multiplier = detail_multiplier
        self.noise_strength = noise_strength
        self.layers = layers
        self.use_vertex_colors = use_vertex_colors
        self.points = []
        self.colors = []
        
        # Check if vertex colors exist
        self.has_vertex_colors = (use_vertex_colors and 
                                  obj.data.vertex_colors and 
                                  len(obj.data.vertex_colors) > 0)
    
    def calculate_curvature_weights(self):
        """Calculate vertex curvature for density mapping"""
        print("Analyzing geometry curvature...")
        
        bm = bmesh.new()
        bm.from_mesh(self.obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        curvatures = []
        for face in bm.faces:
            curvature = 0.0
            face_normal_length = face.normal.length
            
            if face_normal_length < 0.0001:
                curvatures.append(0.0)
                continue
            
            for edge in face.edges:
                for linked_face in edge.link_faces:
                    if linked_face != face:
                        linked_normal_length = linked_face.normal.length
                        if linked_normal_length < 0.0001:
                            continue
                        try:
                            angle = face.normal.angle(linked_face.normal)
                            curvature += angle
                        except:
                            continue
            curvatures.append(curvature)
        
        bm.free()
        
        if len(curvatures) > 0:
            max_curv = max(curvatures) if max(curvatures) > 0 else 1.0
            curvatures = [c / max_curv for c in curvatures]
        
        return curvatures
    
    def get_vertex_color(self, poly, r1, r2, r3):
        """Sample vertex color using barycentric interpolation"""
        if not self.has_vertex_colors:
            return (0.8, 0.8, 0.8)
        
        mesh = self.obj.data
        vc_layer = mesh.vertex_colors.active
        
        # Get the loop indices for this polygon
        loop_indices = list(poly.loop_indices)
        
        if len(loop_indices) < 3:
            return (0.8, 0.8, 0.8)
        
        # Get vertex colors at the three vertices
        color1 = vc_layer.data[loop_indices[0]].color
        color2 = vc_layer.data[loop_indices[1]].color
        color3 = vc_layer.data[loop_indices[2]].color
        
        # Interpolate color using barycentric coordinates
        r = color1[0] * r1 + color2[0] * r2 + color3[0] * r3
        g = color1[1] * r1 + color2[1] * r2 + color3[1] * r3
        b = color1[2] * r1 + color2[2] * r2 + color3[2] * r3
        
        return (r, g, b)
    
    def get_material_color(self, face_idx):
        """Fallback: get material base color for face"""
        mesh = self.obj.data
        
        if not self.obj.material_slots:
            return (0.8, 0.8, 0.8)
        
        poly = mesh.polygons[face_idx]
        if poly.material_index >= len(self.obj.material_slots):
            return (0.8, 0.8, 0.8)
        
        mat = self.obj.material_slots[poly.material_index].material
        if not mat:
            return (0.8, 0.8, 0.8)
        
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    base_color = node.inputs['Base Color'].default_value
                    return tuple(base_color[:3])
        
        return tuple(mat.diffuse_color[:3])
    
    def generate_base_layer(self, curvatures):
        """Generate base point distribution with curvature weighting"""
        print(f"Generating base layer ({self.base_points} points)...")
        
        mesh = self.obj.data
        polygons = mesh.polygons
        
        # Calculate weighted face areas
        weighted_areas = []
        total_weighted_area = 0
        
        for i, poly in enumerate(polygons):
            area = poly.area
            weight = 1.0 + (curvatures[i] * self.detail_multiplier)
            weighted_area = area * weight
            weighted_areas.append(weighted_area)
            total_weighted_area += weighted_area
        
        # Use cumulative distribution for fast sampling
        cumulative_areas = np.cumsum(weighted_areas)
        
        points_generated = 0
        target = self.base_points
        
        print(f"  Starting point generation...")
        
        while points_generated < target:
            # Fast weighted random selection
            rand_val = random.random() * total_weighted_area
            face_idx = np.searchsorted(cumulative_areas, rand_val)
            poly = polygons[face_idx]
            
            # Generate random barycentric coordinates
            r1 = random.random()
            r2 = random.random()
            
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            r3 = 1 - r1 - r2
            
            # Get vertex positions
            verts = [mesh.vertices[v].co for v in poly.vertices]
            
            # Calculate point position
            if len(verts) == 3:
                point = verts[0] * r1 + verts[1] * r2 + verts[2] * r3
            elif len(verts) == 4:
                if random.random() < 0.5:
                    point = verts[0] * r1 + verts[1] * r2 + verts[2] * r3
                else:
                    point = verts[0] * r1 + verts[2] * r2 + verts[3] * r3
            else:
                point = verts[0] * r1 + verts[1] * r2 + verts[2] * r3
            
            self.points.append(point.copy())
            
            # Sample color - use vertex colors if available, else material color
            if self.has_vertex_colors:
                color = self.get_vertex_color(poly, r1, r2, r3)
            else:
                color = self.get_material_color(face_idx)
            
            self.colors.append(color)
            points_generated += 1
            
            if points_generated % 100000 == 0:
                print(f"  Generated {points_generated}/{target} points...")
    
    def add_detail_layers(self, curvatures):
        """Add additional detail layers in high-curvature areas"""
        for layer in range(1, self.layers):
            detail_points = int(self.base_points * (0.5 ** layer))
            print(f"Adding detail layer {layer} ({detail_points} points)...")
            
            mesh = self.obj.data
            polygons = mesh.polygons
            
            detail_weights = []
            total_weight = 0
            
            for i, poly in enumerate(polygons):
                weight = poly.area * (curvatures[i] ** (layer + 1))
                detail_weights.append(weight)
                total_weight += weight
            
            if total_weight == 0:
                continue
            
            cumulative_weights = np.cumsum(detail_weights)
            
            for _ in range(detail_points):
                rand_val = random.random() * total_weight
                face_idx = np.searchsorted(cumulative_weights, rand_val)
                poly = polygons[face_idx]
                
                r1 = random.random()
                r2 = random.random()
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                r3 = 1 - r1 - r2
                
                verts = [mesh.vertices[v].co for v in poly.vertices]
                if len(verts) >= 3:
                    point = verts[0] * r1 + verts[1] * r2 + verts[2] * r3
                    self.points.append(point.copy())
                    
                    # Sample color
                    if self.has_vertex_colors:
                        color = self.get_vertex_color(poly, r1, r2, r3)
                    else:
                        color = self.get_material_color(face_idx)
                    
                    self.colors.append(color)
    
    def apply_noise(self):
        """Add organic variation to point positions"""
        print("Applying organic noise...")
        for i in range(len(self.points)):
            noise_vec = Vector((
                (random.random() - 0.5) * self.noise_strength,
                (random.random() - 0.5) * self.noise_strength,
                (random.random() - 0.5) * self.noise_strength
            ))
            self.points[i] += noise_vec
    
    def export_ply(self, filepath):
        """Export pointcloud as binary PLY with RGB colors"""
        print(f"Exporting to {filepath}...")
        
        with open(filepath, 'wb') as f:
            header = f"""ply
format binary_little_endian 1.0
element vertex {len(self.points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
            f.write(header.encode('ascii'))
            
            for i, point in enumerate(self.points):
                # Position
                f.write(np.array([point.x, point.y, point.z], dtype=np.float32).tobytes())
                
                # Color
                if i < len(self.colors):
                    r = int(max(0, min(1, self.colors[i][0])) * 255)
                    g = int(max(0, min(1, self.colors[i][1])) * 255)
                    b = int(max(0, min(1, self.colors[i][2])) * 255)
                else:
                    r = g = b = 200
                
                f.write(bytes([r, g, b]))
        
        print(f"✓ Exported {len(self.points)} points with RGB colors")
    
    def generate(self):
        """Main generation pipeline"""
        print("=" * 50)
        print("ADAPTIVE POINTCLOUD GENERATION")
        print("=" * 50)
        
        curvatures = self.calculate_curvature_weights()
        self.generate_base_layer(curvatures)
        self.add_detail_layers(curvatures)
        self.apply_noise()
        
        print("=" * 50)
        print(f"✓ COMPLETE: {len(self.points)} total points")
        print("=" * 50)


# ============================================
# MAIN EXECUTION
# ============================================

def generate_adaptive_pointcloud_with_baking():
    """
    Complete workflow: Bake textures → Generate pointcloud → Export
    """
    
    # ===== CONFIGURATION =====
    BASE_POINTS = 1000000        # Base layer point count
    DETAIL_MULTIPLIER = 8.0      # Density multiplier for curved areas
    NOISE_STRENGTH = 0.0005      # Micro-variation
    LAYERS = 3                   # Detail passes
    BAKE_TEXTURES = True         # Bake textures to vertex colors first
    EXPORT_PATH = r"C:\path_to_file"
    
    # Get active object
    obj = bpy.context.active_object
    
    if obj is None or obj.type != 'MESH':
        print("ERROR: Please select a mesh object!")
        return
    
    print(f"Processing: {obj.name}")
    print(f"Vertices: {len(obj.data.vertices)}")
    print(f"Faces: {len(obj.data.polygons)}")
    print()
    
    # Bake textures to vertex colors if requested
    has_vertex_colors = False
    if BAKE_TEXTURES:
        has_vertex_colors = bake_texture_to_vertex_colors(obj)
        print()
    
    # Generate pointcloud
    generator = AdaptivePointcloudGenerator(
        obj=obj,
        base_points=BASE_POINTS,
        detail_multiplier=DETAIL_MULTIPLIER,
        noise_strength=NOISE_STRENGTH,
        layers=LAYERS,
        use_vertex_colors=has_vertex_colors
    )
    
    generator.generate()
    generator.export_ply(EXPORT_PATH)
    
    print()
    print("IMPORT TO TOUCHDESIGNER:")
    print(f"1. Point File In TOP → {EXPORT_PATH}")
    print("2. RGB channels will contain texture colors")
    print("3. Adjust Point Scale for desired density")


# Run it
if __name__ == "__main__":
    generate_adaptive_pointcloud_with_baking()
