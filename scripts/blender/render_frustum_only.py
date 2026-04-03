#!/usr/bin/env python3
"""
Blender script to render only camera frustum meshes with white background.
Run this script in Blender after executing blender_ego_static_with_agent_fig2.py

Usage:
    blender your_file.blend --python render_frustum_only.py
    OR
    Open Blender, go to Scripting workspace, open this file, and run it.
"""

import bpy
import os

def render_frustum_only(output_path=None):
    """
    Hide all objects except frustum snapshots and render with white background.
    
    Args:
        output_path: Optional output path for rendered image. 
                     If None, uses Blender's default render output path.
    """
    print("\n" + "="*60)
    print("RENDERING FRUSTUM ONLY")
    print("="*60)
    
    # 1. Find PoseSnapshots collection
    collection_name = None
    for coll in bpy.data.collections:
        if coll.name.startswith("PoseSnapshots_"):
            collection_name = coll.name
            break
    
    if not collection_name:
        print("ERROR: PoseSnapshots collection not found!")
        print("Available collections:")
        for coll in bpy.data.collections:
            print(f"  - {coll.name}")
        return False
    
    print(f"Found collection: {collection_name}")
    collection = bpy.data.collections.get(collection_name)
    
    # 2. Hide all objects first
    hidden_count = 0
    for obj in bpy.data.objects:
        if not obj.hide_render:
            obj.hide_render = True
            hidden_count += 1
    
    print(f"Hidden {hidden_count} objects")
    
    # 3. Show only frustum snapshots
    frustum_objects = []
    if collection:
        for obj in collection.objects:
            # Show frustum snapshots (Frustum in name)
            if "Frustum" in obj.name:
                obj.hide_render = False
                obj.hide_viewport = False
                frustum_objects.append(obj.name)
                print(f"  Showing: {obj.name}")
    
    if not frustum_objects:
        print("WARNING: No frustum snapshots found in collection!")
        print("Objects in collection:")
        for obj in collection.objects:
            print(f"  - {obj.name}")
        return False
    
    print(f"Found {len(frustum_objects)} frustum snapshot objects")
    
    # 4. Also show original frustum object if it exists (optional)
    original_frustum = bpy.data.objects.get("CameraFrustum")
    if original_frustum:
        original_frustum.hide_render = False
        original_frustum.hide_viewport = False
        print(f"  Also showing original: CameraFrustum")
    
    # 5. Set white background - completely replace World shader
    scene = bpy.context.scene
    world = scene.world
    
    if not world:
        world = bpy.data.worlds.new("World")
        scene.world = world
    
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    
    # Remove ALL existing links first to avoid reference errors
    print("Removing all existing World shader links...")
    for link in list(links):
        links.remove(link)
    
    # Remove ALL existing nodes to clear any image textures or other backgrounds
    print("Removing all existing World shader nodes...")
    node_names = []
    for node in list(nodes):
        try:
            node_names.append(node.name)
            nodes.remove(node)
        except ReferenceError:
            # Node was already removed, skip
            pass
        except Exception as e:
            print(f"  Warning: Could not remove node {node.name}: {e}")
    
    if node_names:
        print(f"  Removed {len(node_names)} nodes: {', '.join(node_names)}")
    
    # Force update to clear any stale references
    bpy.context.view_layer.update()
    
    # Create fresh Background and Output nodes
    bg_node = nodes.new("ShaderNodeBackground")
    output_node = nodes.new("ShaderNodeOutputWorld")
    
    # Set white background - use normal strength to avoid affecting frustum meshes
    # Light bounces are set to 0, so high strength won't affect frustum via indirect lighting
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0  # Normal strength to avoid affecting objects
    
    # Connect Background to Output
    try:
        links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
    except Exception as e:
        print(f"Warning: Could not connect nodes: {e}")
        # Try alternative connection method
        if "Surface" in output_node.inputs:
            output_node.inputs["Surface"].default_value = (1.0, 1.0, 1.0, 1.0)
    
    print("Set white background (all previous nodes removed)")
    
    # 5.5. Enable Film Transparent for alpha channel (transparent background)
    scene.render.film_transparent = True
    print("Enabled Film Transparent (background will be transparent with alpha channel)")
    
    # 5.6. Reduce indirect lighting from World shader to prevent frustum from brightening
    if scene.cycles:
        # Limit bounces to reduce indirect lighting effects
        scene.cycles.max_bounces = 0  # No bounces = no indirect lighting from background
        scene.cycles.diffuse_bounces = 0
        scene.cycles.glossy_bounces = 0
        scene.cycles.transparent_max_bounces = 0
        print("Reduced light bounces to prevent background from affecting frustum")
    
    # 6. Set render output path
    if output_path:
        scene.render.filepath = output_path
        print(f"Output path: {output_path}")
    else:
        # Use default or create frustum_output directory
        blend_filepath = bpy.data.filepath
        if blend_filepath:
            blend_dir = os.path.dirname(blend_filepath)
            output_dir = os.path.join(blend_dir, "frustum_output")
            os.makedirs(output_dir, exist_ok=True)
            scene.render.filepath = os.path.join(output_dir, "frustum_only")
        else:
            scene.render.filepath = "//frustum_only"
        print(f"Output path: {scene.render.filepath}")
    
    # 7. Ensure PNG format with transparency (alpha channel for transparent background)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'  # RGBA includes alpha channel
    print("PNG format with RGBA (alpha channel) enabled for transparent background")
    
    # 8. Render
    print("\nRendering...")
    bpy.ops.render.render(write_still=True)
    
    print("\n" + "="*60)
    print(f"✅ RENDERING COMPLETE!")
    print(f"Output: {scene.render.filepath}.png")
    print(f"Rendered {len(frustum_objects)} frustum snapshot objects")
    print("="*60)
    
    return True


def main():
    """Main function - can be customized with command line args if needed."""
    import sys
    
    # Parse command line args if provided
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    
    output_path = None
    if argv and len(argv) > 0:
        output_path = argv[0]
    
    success = render_frustum_only(output_path)
    
    if not success:
        print("\nERROR: Rendering failed. Check the messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    # If run directly in Blender, execute main
    main()

