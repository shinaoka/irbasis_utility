function(add_python_test test)
    add_test(NAME python_${test} COMMAND ${PYTHON_EXECUTABLE} ${test}.py)
    set_tests_properties(python_${test} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/test/python")
endfunction(add_python_test)

find_package(PythonInterp REQUIRED)

# Copy all library files
file(GLOB_RECURSE LIB_FILES RELATIVE ${CMAKE_SOURCE_DIR}/python "${CMAKE_SOURCE_DIR}/python/*.py")
foreach(libfile ${LIB_FILES})
    configure_file(${CMAKE_SOURCE_DIR}/python/${libfile} ${CMAKE_BINARY_DIR}/test/python/irbasis_util/${libfile} COPYONLY)
endforeach()

# copy test files
file(GLOB_RECURSE TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
foreach(testfile ${TEST_FILES})
    configure_file(${testfile} ${CMAKE_BINARY_DIR}/test/python COPYONLY)
endforeach()

set(python_test_src internal_test two_point_basis_test three_point_test four_point_test four_point_ph_view_test sampling_test regression_test tensor_regression_test)
foreach(test ${python_test_src})
    add_python_test(${test})
endforeach(test)
