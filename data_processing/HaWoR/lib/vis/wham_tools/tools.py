import numpy as np


def checkerboard_geometry(
    length=12.0,
    color0=[172/255, 172/255, 172/255],
    color1=[215/255, 215/255, 215/255],
    tile_width=0.5,
    alpha=1.0,
    up="y",
    c1=0.0,
    c2=0.0,
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = max(2, int(length / tile_width))
    vertices = []
    vert_colors = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 2] += c2
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 1] += c2

            cur_faces = np.array(
                [[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64
            )
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.float32)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors