function(AssignSourceGroup)
    foreach (_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else ()
            set(_source_rel "${_source}")
        endif ()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach ()
endfunction(AssignSourceGroup)

macro(AddTestProgram TestFile Libraries Category)
    get_filename_component(FILE_NAME ${TestFile} NAME_WE)
    add_executable(${FILE_NAME} ${TestFile})

    set_target_properties(${FILE_NAME} PROPERTIES FOLDER "Tests/${Category}")

    target_link_libraries(${FILE_NAME} PUBLIC ${Libraries})
#    set_target_properties(${FILE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${SHARK_BINARY_DIR})
    add_test(NAME "${FILE_NAME}Test" COMMAND ${FILE_NAME}) # WORKING_DIRECTORY ${SHARK_BINARY_DIR}
endmacro()

macro(AddBenchProgram BenchFile Libraries Category)
    get_filename_component(FILE_NAME ${BenchFile} NAME_WE)
    add_executable(${FILE_NAME} ${BenchFile})

    set_target_properties(${FILE_NAME} PROPERTIES FOLDER "Benchmarks/${Category}")

    SetDefaultCompileDefinitions(${FILE_NAME})
    target_link_libraries(${FILE_NAME} PRIVATE ${Libraries})
#    set_target_properties(${FILE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${SHARK_BINARY_DIR})
endmacro()
