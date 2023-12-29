add_library(llava STATIC
            llava.cpp
            llava.h
            clip.cpp
            clip.h
            )

target_link_libraries(llava PUBLIC ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(llava PUBLIC .)
target_include_directories(llava PUBLIC ../..)
target_include_directories(llava PUBLIC ../../common)

target_compile_features(llava PRIVATE cxx_std_11)

if(UNIX OR APPLE)
    target_compile_options(llava PRIVATE -Wno-cast-qual)
endif()