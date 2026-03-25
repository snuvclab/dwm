#!/usr/bin/env python3

import bpy


class StaticSceneController:
    def __init__(self):
        self._original_animation_state = {
            "camera_parent": None,
            "actions": {},
        }

    def sample_camera_world_transforms(self, camera_obj, frames):
        scene = bpy.context.scene
        locations = []
        rotations = []
        previous_frame = scene.frame_current
        try:
            for frame in frames:
                scene.frame_set(frame)
                matrix_world = camera_obj.matrix_world.copy()
                locations.append(matrix_world.to_translation())
                rotations.append(matrix_world.to_quaternion())
        finally:
            scene.frame_set(previous_frame)
        return locations, rotations

    def disable_animations_except_camera(self, camera_obj):
        self._original_animation_state = {"camera_parent": None, "actions": {}}

        if camera_obj.parent is not None:
            self._original_animation_state["camera_parent"] = (
                camera_obj.parent,
                camera_obj.parent_type,
                camera_obj.parent_bone,
                camera_obj.matrix_world.copy(),
            )
            bpy.context.view_layer.objects.active = camera_obj
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

        for obj in bpy.data.objects:
            if obj == camera_obj or not obj.animation_data:
                continue
            self._original_animation_state["actions"][obj.name] = obj.animation_data.action
            if obj.animation_data.action:
                obj.animation_data.action.use_fake_user = True
            obj.animation_data.action = None

        bpy.context.view_layer.update()

    def restore_animations(self, camera_obj):
        if not self._original_animation_state:
            return

        for obj_name, action in self._original_animation_state.get("actions", {}).items():
            obj = bpy.data.objects.get(obj_name)
            if not obj:
                continue
            if not obj.animation_data:
                obj.animation_data_create()
            obj.animation_data.action = action

        camera_parent = self._original_animation_state.get("camera_parent")
        if camera_parent is not None:
            parent, parent_type, parent_bone, matrix_world_before = camera_parent
            camera_obj.parent = parent
            camera_obj.parent_type = parent_type
            if parent_type == "BONE":
                camera_obj.parent_bone = parent_bone
            camera_obj.matrix_world = matrix_world_before

        self._original_animation_state = {"camera_parent": None, "actions": {}}
        bpy.context.view_layer.update()

    def bake_camera_keys(self, camera_obj, frames, locations, rotations):
        if not camera_obj.animation_data:
            camera_obj.animation_data_create()
        if not camera_obj.animation_data.action:
            camera_obj.animation_data.action = bpy.data.actions.new(name="POV_Camera_Baked")
        action = camera_obj.animation_data.action

        for fcurve in list(action.fcurves):
            action.fcurves.remove(fcurve)

        location_curves = [action.fcurves.new(data_path="location", index=i) for i in range(3)]
        rotation_curves = [
            action.fcurves.new(data_path="rotation_quaternion", index=i) for i in range(4)
        ]

        camera_obj.rotation_mode = "QUATERNION"
        for frame, location, rotation in zip(frames, locations, rotations):
            camera_obj.location = location
            camera_obj.rotation_quaternion = rotation
            for index, curve in enumerate(location_curves):
                curve.keyframe_points.insert(
                    frame=frame,
                    value=camera_obj.location[index],
                    options={"FAST"},
                )
            for index, curve in enumerate(rotation_curves):
                curve.keyframe_points.insert(
                    frame=frame,
                    value=camera_obj.rotation_quaternion[index],
                    options={"FAST"},
                )

        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"

    def clear_camera_keys(self, camera_obj):
        if camera_obj.animation_data and camera_obj.animation_data.action:
            action = camera_obj.animation_data.action
            camera_obj.animation_data_clear()
            try:
                bpy.data.actions.remove(action)
            except Exception:
                pass

