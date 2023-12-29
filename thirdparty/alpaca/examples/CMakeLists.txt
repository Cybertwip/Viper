find_package(Threads REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(llava)
    add_subdirectory(alpaca)
endif()
