import trimesh

mesh = trimesh.load('suzanne.obj')

scene = trimesh.Scene([mesh])
scene.show()