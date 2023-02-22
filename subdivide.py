"""Run this in pvpython to leverage ParaView's GMsh reader/writer."""
# trace generated using paraview version 5.11.0-RC1
from paraview.simple import *

# load plugin
LoadPlugin(
    "/Applications/ParaView-5.11.0-RC1.app/Contents/MacOS/../Plugins/GmshIO.so",
    remote=False,
    ns=globals(),
)

# create a new 'Legacy VTK Reader'
a2022_11_30_11_52_49_mesh_geometryvtk = LegacyVTKReader(
    registrationName="2022_11_30_11_52_49_mesh_geometry.vtk",
    FileNames=[
        "/Users/bane.sullivan/Software/Harvard/kale/2022_11_30_11_52_49_mesh_geometry.vtk"
    ],
)

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(
    registrationName="ExtractSurface1", Input=a2022_11_30_11_52_49_mesh_geometryvtk
)

# create a new 'Subdivide'
subdivide1 = Subdivide(registrationName="Subdivide1", Input=extractSurface1)

# Properties modified on subdivide1
subdivide1.NumberofSubdivisions = 3

# create a new 'Append Datasets'
appendDatasets1 = AppendDatasets(registrationName="AppendDatasets1", Input=subdivide1)

# save data
SaveData(
    "/Users/bane.sullivan/Software/Harvard/kale/foo.msh",
    proxy=appendDatasets1,
    WriteGmshSpecificArray=1,
)
