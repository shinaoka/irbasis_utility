# Copy files into a build directories
set(script_files __init__.py two_point_basis.py three_point_ph.py regression.py)
foreach(file ${script_files})
    configure_file(${CMAKE_SOURCE_DIR}/python/${file} ${CMAKE_BINARY_DIR}/irbasis_utility/${file} COPYONLY)
endforeach(file)
configure_file(${CMAKE_SOURCE_DIR}/python/setup.py ${CMAKE_BINARY_DIR}/setup.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/setup.cfg ${CMAKE_BINARY_DIR}/setup.cfg COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/README.md ${CMAKE_BINARY_DIR}/README.md COPYONLY)
