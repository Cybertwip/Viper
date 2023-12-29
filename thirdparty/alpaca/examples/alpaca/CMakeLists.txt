set(TARGET alpaca)
option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" OFF)
add_library(${TARGET} alpaca.hpp alpaca.cpp json.hpp)
target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)
target_link_libraries(${TARGET} PUBLIC common llama llava ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
