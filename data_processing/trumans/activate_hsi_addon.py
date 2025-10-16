# activate_hsi_addon.py
print("Run 'blender -b --python activate_hsi_addon.py'")
import bpy

addon_name = "HSI_addon-zzy" # This is the folder name of your addon

try:
    # Enable the addon
    bpy.ops.preferences.addon_enable(module=addon_name)
    print(f"Add-on '{addon_name}' enabled successfully.")

    # Save user preferences (important for persistent activation)
    bpy.ops.wm.save_userpref()
    print("User preferences saved, add-on activation is persistent.")

except Exception as e:
    print(f"Error enabling or saving add-on '{addon_name}': {e}")

# You might want to exit Blender after this, or perform other headless tasks.
# bpy.ops.wm.quit_blender() # Use with caution, it will quit the Blender process
