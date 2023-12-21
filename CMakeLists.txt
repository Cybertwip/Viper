cmake_minimum_required(VERSION 3.13)

project(viper C CXX)

if(XCODE)
  set(CMAKE_BUILD_TYPE Debug)
else()
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib CACHE STRING "Shared Libraries")

file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" piper_version)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  # Force compiler to use UTF-8 for IPA constants
  add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
elseif(NOT APPLE)
  # Linux flags
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wl,-rpath,'$ORIGIN'")
  string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

add_library(viper SHARED src/cpp/viper.hpp src/cpp/viper.cpp src/cpp/piper.cpp)
add_executable(main src/cpp/main.cpp)

# ---- piper-phonemize ---
add_subdirectory(thirdparty/fmt)
add_subdirectory(thirdparty/spdlog)
add_subdirectory(thirdparty/piper_phonemize)

target_link_libraries(viper
  PUBLIC
  piper_phonemize
  fmt
  spdlog
  ${PIPER_EXTRA_LIBRARIES}
)
target_link_libraries(main
PUBLIC
viper)

target_include_directories(viper PUBLIC thirdparty/spdlog/include)

target_compile_definitions(viper PUBLIC _PIPER_VERSION=${piper_version})

set_property(TARGET viper PROPERTY DEFINE_SYMBOL "")


file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/cpp/viper.hpp DESTINATION ${CMAKE_CURRENT_LIST_DIR}/include)
