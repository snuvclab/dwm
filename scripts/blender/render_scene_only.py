#!/usr/bin/env python3
"""
Blender script to render only scene objects (excluding character and frustum) with transparent background.
Run this script in Blender after executing blender_ego_static_with_agent_fig2.py

Usage:
    blender your_file.blend --python render_scene_only.py
    OR
    Open Blender, go to Scripting workspace, open this file, and run it.
"""

import bpy
import os

def render_scene_only(output_path=None):
    """
    Hide character and frustum objects, render only scene with transparent background.
    
    Args:
        output_path: Optional output path for rendered image. 
                     If None, uses Blender's default render output path.
    """
    print("\n" + "="*60)
    print("RENDERING SCENE ONLY (excluding character and frustum)")
    print("="*60)
    
    # 1. Find PoseSnapshots collection (to hide snapshots)
    collection_name = None
    for coll in bpy.data.collections:
        if coll.name.startswith("PoseSnapshots_"):
            collection_name = coll.name
            break
    
    if collection_name:
        print(f"Found PoseSnapshots collection: {collection_name}")
        collection = bpy.data.collections.get(collection_name)
        if collection:
            # Hide all snapshot objects
            for obj in collection.objects:
                obj.hide_render = True
                obj.hide_viewport = False
                print(f"  Hidden snapshot: {obj.name}")
    
    # 2. Hide character-related objects
    character_objects = [
        "CC_Base_Body",
        "CC_Hand_L",
        "CC_Hand_R",
        "CC_Base_Eye",
        "POV_Camera",  # POV camera is part of character setup
    ]
    
    # Find armature
    armature_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and 'CC_Base_Hip' in obj.pose.bones:
            armature_obj = obj
            character_objects.append(obj.name)
            break
    
    # Hide character objects
    hidden_character = []
    for obj_name in character_objects:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.hide_render = True
            obj.hide_viewport = False
            hidden_character.append(obj_name)
            print(f"  Hidden character object: {obj_name}")
    
    # Hide all children of armature
    if armature_obj:
        for child in armature_obj.children:
            child.hide_render = True
            child.hide_viewport = False
            print(f"  Hidden armature child: {child.name}")
    
    # 3. Hide frustum objects
    frustum_objects = []
    for obj in bpy.data.objects:
        if "Frustum" in obj.name or "CameraFrustum" in obj.name:
            obj.hide_render = True
            obj.hide_viewport = False
            frustum_objects.append(obj.name)
            print(f"  Hidden frustum: {obj.name}")
    
    # 4. Show all other objects (scene objects)
    scene_objects = []
    for obj in bpy.data.objects:
        # Skip cameras and lights (keep them visible for scene lighting)
        if obj.type in {'CAMERA', 'LIGHT'}:
            # Keep render camera visible, but hide POV camera
            if obj.name == "Camera":
                obj.hide_render = False
                obj.hide_viewport = False
            continue
        
        # Skip hidden character/frustum objects
        if obj.name in character_objects or obj.name in frustum_objects:
            continue
        
        # Skip if it's a child of armature
        if armature_obj and obj.parent == armature_obj:
            continue
        
        # Skip if it's in PoseSnapshots collection
        if collection and obj.name in [o.name for o in collection.objects]:
            continue
        
        # Show scene objects
        if obj.hide_render:
            obj.hide_render = False
            obj.hide_viewport = False
            scene_objects.append(obj.name)
            print(f"  Showing scene object: {obj.name}")
    
    print(f"\nFound {len(scene_objects)} scene objects to render")
    print(f"Hidden {len(hidden_character)} character objects")
    print(f"Hidden {len(frustum_objects)} frustum objects")
    
    # 5. Set transparent background - completely replace World shader
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
    print("\nRemoving all existing World shader links...")
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
    
    # Set background - use higher strength for better scene lighting
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 3.0  # Higher strength for better scene illumination
    
    # Connect Background to Output
    try:
        links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
    except Exception as e:
        print(f"Warning: Could not connect nodes: {e}")
        # Try alternative connection method
        if "Surface" in output_node.inputs:
            output_node.inputs["Surface"].default_value = (1.0, 1.0, 1.0, 1.0)
    
    print("Set World shader (all previous nodes removed)")
    
    # 5.5. Enable Film Transparent for alpha channel (transparent background)
    scene.render.film_transparent = True
    print("Enabled Film Transparent (background will be transparent with alpha channel)")
    
    # 5.6. Enable proper lighting bounces for scene objects
    if scene.cycles:
        # Enable bounces for proper scene lighting (scene objects need indirect lighting)
        scene.cycles.max_bounces = 4  # Allow bounces for proper scene illumination
        scene.cycles.diffuse_bounces = 4
        scene.cycles.glossy_bounces = 4
        scene.cycles.transparent_max_bounces = 8
        print("Enabled light bounces for proper scene object illumination")
    
    # 6. Set render output path
    if output_path:
        scene.render.filepath = output_path
        print(f"Output path: {output_path}")
    else:
        # Use default or create scene_output directory
        blend_filepath = bpy.data.filepath
        if blend_filepath:
            blend_dir = os.path.dirname(blend_filepath)
            output_dir = os.path.join(blend_dir, "scene_output")
            os.makedirs(output_dir, exist_ok=True)
            scene.render.filepath = os.path.join(output_dir, "scene_only")
        else:
            scene.render.filepath = "//scene_only"
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
    print(f"Rendered {len(scene_objects)} scene objects")
    print(f"Hidden {len(hidden_character)} character objects")
    print(f"Hidden {len(frustum_objects)} frustum objects")
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
    
    success = render_scene_only(output_path)
    
    if not success:
        print("\nERROR: Rendering failed. Check the messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    # If run directly in Blender, execute main
    main()

