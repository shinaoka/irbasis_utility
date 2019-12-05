# Copy files into a build directories
set(script_files __init__.py two_point_basis.py three_point.py four_point.py
four_point_ph_view.py regression.py internal.py tensor_regression.py lsqr.py
tensor_regression_mps.py tensor_network.py auto_als.py tensor_regression_auto_als.py)
foreach(file ${script_files})
    configure_file(${CMAKE_SOURCE_DIR}/python/${file} ${CMAKE_BINARY_DIR}/irbasis_util/${file} COPYONLY)
endforeach(file)
configure_file(${CMAKE_SOURCE_DIR}/python/setup.py ${CMAKE_BINARY_DIR}/setup.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/setup.cfg ${CMAKE_BINARY_DIR}/setup.cfg COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/README.md ${CMAKE_BINARY_DIR}/README.md COPYONLY)
