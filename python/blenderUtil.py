import numpy as np


def mapToVertices(noise_map, max_height):

    width, height = noise_map.shape

    xmin, ymin = -width/2, -height/2
    cur_index = 0

    raw_vertices = []
    raw_faces = []
    raw_edges = []


    for x in range(width):
        for y in range(height):
            
            # For every height
            for z in range(noise_map[x,y]):

                # Generate 8 vertices
                v0 = (xmin + float(x), ymin + float(y), float(z))
                v1 = (xmin + float(x+1), ymin + float(y), float(z))
                v2 = (xmin + float(x), ymin + float(y+1), float(z))
                v3 = (xmin + float(x+1), ymin + float(y+1), float(z))

                v4 = (xmin + float(x), ymin + float(y), float(z+1))
                v5 = (xmin + float(x+1), ymin + float(y), float(z+1))
                v6 = (xmin + float(x), ymin + float(y+1), float(z+1))
                v7 = (xmin + float(x+1), ymin + float(y+1), float(z+1))

                raw_vertices.append(v0)
                raw_vertices.append(v1)
                raw_vertices.append(v2)
                raw_vertices.append(v3)
                raw_vertices.append(v4)
                raw_vertices.append(v5)
                raw_vertices.append(v6)
                raw_vertices.append(v7)

                # Generate 6 faces

                # v0 -> v1 -> v3 -> v2
                f0 = [cur_index, cur_index + 1, cur_index + 2, cur_index + 3]
                # v0 -> v1 -> v5 -> v4
                f1 = [cur_index, cur_index + 1, cur_index + 5, cur_index + 4]
                # v0 -> v2 -> v6 -> v4
                f2 = [cur_index, cur_index + 2, cur_index + 6, cur_index + 4]
                # v3 -> v2 -> v6 -> v7
                f3 = [cur_index + 3, cur_index + 2, cur_index + 6, cur_index + 7]
                # v3 -> v1 -> v5 -> v7
                f4 = [cur_index + 3, cur_index + 1, cur_index + 5, cur_index + 7]
                # v4 -> v5 -> v7 -> v6
                f5 = [cur_index + 4, cur_index + 5, cur_index + 7, cur_index + 6]

                cur_index += 8
                raw_faces.append(f0)
                raw_faces.append(f1)
                raw_faces.append(f2)
                raw_faces.append(f3)
                raw_faces.append(f4)
                raw_faces.append(f5)


                # Generate 12 edges


    return raw_vertices, raw_faces


def generateRandMap(length, max_height):

    return np.random.randint(0, max_height, size=(length, length), dtype="int")
