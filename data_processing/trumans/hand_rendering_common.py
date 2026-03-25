#!/usr/bin/env python3

import math

import bpy
import numpy as np
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRendererWithFragments,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes


BODY_NAME = "CC_Base_Body"


def create_phong_material(name, color):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    node_tree = material.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)
    output = node_tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.3
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.8
    elif "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.8
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.45
    node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def setup_simple_hand_rendering():
    body = bpy.data.objects.get(BODY_NAME)
    if not body:
        print(f"Error: {BODY_NAME} not found")
        return False

    left_hand_groups = [
        "CC_Base_L_Thumb1", "CC_Base_L_Thumb2", "CC_Base_L_Thumb3",
        "CC_Base_L_Index1", "CC_Base_L_Index2", "CC_Base_L_Index3",
        "CC_Base_L_Mid1", "CC_Base_L_Mid2", "CC_Base_L_Mid3",
        "CC_Base_L_Ring1", "CC_Base_L_Ring2", "CC_Base_L_Ring3",
        "CC_Base_L_Pinky1", "CC_Base_L_Pinky2", "CC_Base_L_Pinky3",
        "CC_Base_L_Hand",
    ]
    right_hand_groups = [
        "CC_Base_R_Thumb1", "CC_Base_R_Thumb2", "CC_Base_R_Thumb3",
        "CC_Base_R_Index1", "CC_Base_R_Index2", "CC_Base_R_Index3",
        "CC_Base_R_Mid1", "CC_Base_R_Mid2", "CC_Base_R_Mid3",
        "CC_Base_R_Ring1", "CC_Base_R_Ring2", "CC_Base_R_Ring3",
        "CC_Base_R_Pinky1", "CC_Base_R_Pinky2", "CC_Base_R_Pinky3",
        "CC_Base_R_Hand",
    ]

    for name in ("CC_Hand_L", "CC_Hand_R"):
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    def duplicate_object(src_obj, new_name):
        dup = src_obj.copy()
        dup.data = src_obj.data.copy()
        dup.animation_data_clear()
        dup.name = new_name
        bpy.context.scene.collection.objects.link(dup)
        dup.modifiers.clear()
        for modifier in src_obj.modifiers:
            if modifier.type == "ARMATURE":
                dup_modifier = dup.modifiers.new(modifier.name, modifier.type)
                dup_modifier.object = modifier.object
                if hasattr(dup_modifier, "use_deform_preserve_volume"):
                    dup_modifier.use_deform_preserve_volume = getattr(
                        modifier, "use_deform_preserve_volume", True
                    )
        return dup

    hand_l = duplicate_object(body, "CC_Hand_L")
    hand_r = duplicate_object(body, "CC_Hand_R")

    def ensure_vgroup(obj, name):
        group = obj.vertex_groups.get(name)
        if group is None:
            group = obj.vertex_groups.new(name=name)
        return group

    def build_union_via_modifiers(obj, target_name, source_group_names):
        ensure_vgroup(obj, target_name)
        for group_name in source_group_names:
            if obj.vertex_groups.get(group_name) is None:
                continue
            modifier = obj.modifiers.new(
                name=f"VWM_{target_name}_ADD_{group_name}",
                type="VERTEX_WEIGHT_MIX",
            )
            modifier.vertex_group_a = target_name
            modifier.vertex_group_b = group_name
            modifier.mix_mode = "ADD"
            modifier.mix_set = "ALL"
            modifier.mask_constant = 1.0

    build_union_via_modifiers(hand_l, "Hand_L_All", left_hand_groups)
    build_union_via_modifiers(hand_r, "Hand_R_All", right_hand_groups)

    def add_mask(obj, group_name):
        modifier = obj.modifiers.new(name=f"Mask_{group_name}", type="MASK")
        modifier.vertex_group = group_name
        modifier.invert_vertex_group = False
        modifier.show_viewport = True
        modifier.show_render = True

    add_mask(hand_l, "Hand_L_All")
    add_mask(hand_r, "Hand_R_All")

    mat_l = create_phong_material("LeftHandMaterial", (0.0, 1.0, 0.0))
    mat_r = create_phong_material("RightHandMaterial", (1.0, 0.0, 0.0))
    for obj, material in ((hand_l, mat_l), (hand_r, mat_r)):
        obj.data.materials.clear()
        obj.data.materials.append(material)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

    body.hide_viewport = True
    body.hide_render = True
    print("Built hand-only objects: CC_Hand_L / CC_Hand_R.")
    return True


def hide_non_hand_objects():
    keep = {"CC_Hand_L", "CC_Hand_R"}
    for obj in bpy.data.objects:
        if obj.name in keep or obj.type in {"CAMERA", "LIGHT"}:
            obj.hide_viewport = False
            obj.hide_render = False
        else:
            obj.hide_viewport = True
            obj.hide_render = True


def setup_lighting_for_hands():
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputWorld")
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.08, 0.08, 0.08, 1.0)
    background.inputs["Strength"].default_value = 1.5
    links.new(background.outputs["Background"], output.inputs["Surface"])


def get_hand_mesh_data(hand_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = hand_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        matrix_world = hand_obj.matrix_world
        verts = []
        for vertex in eval_mesh.vertices:
            world = matrix_world @ vertex.co
            verts.append([world.x, world.y, world.z])

        faces = []
        for poly in eval_mesh.polygons:
            if len(poly.vertices) == 3:
                i0, i1, i2 = poly.vertices
                faces.append([i0, i1, i2])
            elif len(poly.vertices) == 4:
                i0, i1, i2, i3 = poly.vertices
                faces.append([i0, i1, i2])
                faces.append([i0, i2, i3])

        return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)
    finally:
        eval_obj.to_mesh_clear()


def get_visible_hand_objects():
    hand_objects = []
    for obj_name in ("CC_Hand_L", "CC_Hand_R"):
        obj = bpy.data.objects.get(obj_name)
        if obj and obj.visible_get():
            hand_objects.append(obj)
    return hand_objects


def _get_pytorch3d_camera_setup(camera_obj, render_shape, device):
    if render_shape is None:
        height, width = 480, 720
    else:
        height, width = render_shape

    fov_rad = camera_obj.data.angle
    fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    focal = torch.tensor([[fx, fy]], device=device, dtype=torch.float32)
    principal = torch.tensor([[cx, cy]], device=device, dtype=torch.float32)
    image_size = torch.tensor([[height, width]], device=device, dtype=torch.int64)
    cameras = PerspectiveCameras(
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, device=device),
        focal_length=focal,
        principal_point=principal,
        in_ndc=False,
        image_size=image_size,
        device=device,
    )
    return height, width, cameras


def _get_camera_world_to_pytorch3d(camera_obj, device):
    matrix_wc = np.array(camera_obj.matrix_world.inverted(), dtype=np.float32)
    rotation_wc = matrix_wc[:3, :3]
    translation_wc = matrix_wc[:3, 3]
    convert = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    rotation_p3d = torch.from_numpy(convert @ rotation_wc).to(device)
    translation_p3d = torch.from_numpy(convert @ translation_wc).to(device)
    return rotation_p3d, translation_p3d


def _matches_target_hand(obj_name, target_hand):
    if target_hand == "left":
        return "CC_Hand_L" in obj_name
    if target_hand == "right":
        return "CC_Hand_R" in obj_name
    return True


class HandDepthRasterizer:
    def __init__(self, camera_obj, render_shape=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height, self.width, cameras = _get_pytorch3d_camera_setup(
            camera_obj,
            render_shape,
            self.device,
        )
        raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
        )
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(self.device)

    @torch.no_grad()
    def render_depth(self, camera_obj, hand_objects, target_hand=None):
        rotation_p3d, translation_p3d = _get_camera_world_to_pytorch3d(camera_obj, self.device)
        verts_list = []
        faces_list = []
        vertex_offset = 0

        for obj in hand_objects:
            if not obj.visible_get() or not _matches_target_hand(obj.name, target_hand):
                continue

            verts_np, faces_np = get_hand_mesh_data(obj)
            if verts_np.size == 0 or faces_np.size == 0:
                continue

            verts_world = torch.from_numpy(verts_np).to(self.device)
            verts_cam = (verts_world @ rotation_p3d.t()) + translation_p3d
            faces = torch.from_numpy(faces_np).to(self.device).long() + vertex_offset
            verts_list.append(verts_cam)
            faces_list.append(faces)
            vertex_offset += verts_cam.shape[0]

        if not verts_list:
            return np.full((self.height, self.width), np.inf, dtype=np.float32)

        mesh = Meshes(
            verts=[torch.cat(verts_list, dim=0)],
            faces=[torch.cat(faces_list, dim=0)],
        )
        fragments = self.rasterizer(mesh)
        zbuf = fragments.zbuf[0, :, :, 0].float()
        valid = fragments.pix_to_face[0, :, :, 0] >= 0
        depth = torch.where(valid, zbuf, torch.full_like(zbuf, float("inf")))
        return depth.cpu().numpy()


def render_hands_depth_pytorch3d(camera_obj, hand_objects, render_shape=None, target_hand=None, renderer=None):
    if renderer is None:
        renderer = HandDepthRasterizer(camera_obj, render_shape=render_shape)
    return renderer.render_depth(camera_obj, hand_objects, target_hand=target_hand)


@torch.no_grad()
def render_hands_pytorch3d(
    camera_obj,
    hand_objects,
    render_shape=None,
    separate_hands=False,
    target_hand=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width, cameras = _get_pytorch3d_camera_setup(camera_obj, render_shape, device)
    rotation_p3d, translation_p3d = _get_camera_world_to_pytorch3d(camera_obj, device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 0.0]],
        ambient_color=((0.7, 0.7, 0.7),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.0, 0.0, 0.0),),
    )
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

    final_rgb = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
    final_z = torch.full((height, width), float("inf"), device=device, dtype=torch.float32)

    for obj in hand_objects:
        if not obj.visible_get():
            continue
        if separate_hands and not _matches_target_hand(obj.name, target_hand):
            continue

        verts_np, faces_np = get_hand_mesh_data(obj)
        if verts_np.size == 0 or faces_np.size == 0:
            continue

        verts_world = torch.from_numpy(verts_np).to(device)
        verts_cam = (verts_world @ rotation_p3d.t()) + translation_p3d
        verts = verts_cam.unsqueeze(0)
        faces = torch.from_numpy(faces_np).to(device).unsqueeze(0).long()

        if "CC_Hand_L" in obj.name:
            color = (0.5, 0.8, 0.5)
        else:
            color = (0.8, 0.4, 0.4)
        tex = torch.tensor(color, device=device).view(1, 1, 3).expand(1, verts.shape[1], 3)
        mesh = Meshes(verts=verts, faces=faces, textures=TexturesVertex(tex))

        images, fragments = renderer(mesh)
        rgb = images[0, :, :, :3]
        zbuf = fragments.zbuf[0, :, :, 0].float()
        valid = fragments.pix_to_face[0, :, :, 0] >= 0

        closer = valid & (zbuf < final_z)
        final_rgb = torch.where(closer.unsqueeze(-1), rgb, final_rgb)
        final_z = torch.where(closer, zbuf, final_z)

    image = (final_rgb.clamp(0, 1) * 255.0).byte().cpu().numpy()
    depth = final_z.cpu().numpy()
    return image, depth
