import bpy

randMap = [[8, 6, 8, 6, 1],
           [9, 1, 2, 2, 7],
           [3, 3, 3, 8, 4],
           [7, 9, 1, 9, 4],
           [3, 9, 1, 1, 3]]

# Add cubes in the scene based on map values
for x in range(5):
    for y in range(5):

        for z in range(randMap[x][y]):
            bpy.ops.mesh.primitive_cube_add(location=(2*x, 2*y, 2*z))

# Remove default cube   
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()


# Select all new cubes, join into one object,
# and rename "Terrain"
parent = None

bpy.ops.object.select_all(action='DESELECT')
for object in bpy.data.objects:
    if "Cube" in object.name:
        object.select_set(True)
        parent = object
        
bpy.context.view_layer.objects.active = parent
bpy.ops.object.join()
bpy.context.active_object.name = "Terrain"


# Hollow out mesh by merging duplicate vertices and 
# deleting inner faces
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold = 0.05)
bpy.ops.object.mode_set(mode='OBJECT')