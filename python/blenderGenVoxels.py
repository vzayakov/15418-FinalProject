import bpy

'''
randMap = [[8, 6, 8, 6, 1],
           [9, 1, 2, 2, 7],
           [3, 3, 3, 8, 4],
           [7, 9, 1, 9, 4],
           [3, 9, 1, 1, 3]]
'''

genMap = list()
fd = open("/Users/victorz/Desktop/15418-FinalProject/noisemap.txt", "r")
next(fd) # Skip the first line
for line in fd:
    linearray = [round((12 * (float(n) + 1))) for n in line.split()]
    genMap.append(linearray)

# Add cubes in the scene based on map values
for x in range(50):
    for y in range(50):

        z = genMap[x][y]
        bpy.ops.mesh.primitive_cube_add(location=(x, y, z))

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


# Scale and place in the center
bpy.context.scene.objects['Terrain'].scale = (1, 1, 1)    # TODO: Get rid of magic numbers
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
bpy.context.scene.objects['Terrain'].location = (0.0, 0.0, 0.0)

# Render using BLENDER_EEVEE for speed or 
# CYCLES for accuracy
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.filepath = '/Users/victorz/Desktop/rendertest.jpg'    # TODO: Get relative path
bpy.ops.render.render(write_still=True)