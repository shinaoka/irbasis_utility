function(add_python_test test)
    add_test(NAME python_${test} COMMAND ${PYTHON_EXECUTABLE} ${test}.py)
    set_tests_properties(python_${test} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/test/python")
endfunction(add_python_test)

find_package(PythonInterp REQUIRED)

# Copy all library files
file(GLOB_RECURSE LIB_FILES "${CMAKE_SOURCE_DIR}/python/*.py")
foreach(libfile ${LIB_FILES})
    configure_file(${libfile} ${CMAKE_BINARY_DIR}/test/python/irbasis_utility/ COPYONLY)
endforeach()

# copy test files
file(GLOB_RECURSE TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
foreach(testfile ${TEST_FILES})
    configure_file(${testfile} ${CMAKE_BINARY_DIR}/test/python COPYONLY)
endforeach()

set(python_test_src  two_point_basis_test three_point_ph_test sampling_test)
foreach(test ${python_test_src})
    add_python_test(${test})
endforeach(test)
