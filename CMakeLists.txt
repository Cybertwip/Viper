cmake_minimum_required(VERSION 3.13)

project(viper C CXX)

if(XCODE)
  set(CMAKE_BUILD_TYPE Debug)
else()
  set(CMAKE_BUILD_TYPE Release)
endif()

if(APPLE)
  set(CMAKE_INSTALL_PREFIX /opt/homebrew)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/artifacts/lib CACHE STRING "Static Libraries")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/artifacts/lib CACHE STRING "Shared Libraries")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/artifacts/bin CACHE STRING "Executables")

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

add_executable(piper src/cpp/main.cpp src/cpp/piper.cpp)
add_library(viper SHARED src/cpp/viper.hpp src/cpp/viper.cpp src/cpp/piper.cpp)

# ---- piper-phonemize ---
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/thirdparty/fmt)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/thirdparty/spdlog)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/thirdparty/piper_phonemize)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/thirdparty/alpaca)

target_link_libraries(viper
  PUBLIC
  alpaca
  piper_phonemize
  fmt
  spdlog
  ${PIPER_EXTRA_LIBRARIES}
)

target_link_libraries(piper
  PUBLIC
  alpaca
  piper_phonemize
  fmt
  spdlog
  ${PIPER_EXTRA_LIBRARIES}
)

target_compile_definitions(viper PUBLIC _PIPER_VERSION=${piper_version})

set_property(TARGET viper PROPERTY DEFINE_SYMBOL "")

file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/cpp/viper.hpp DESTINATION ${CMAKE_CURRENT_LIST_DIR}/include)


set_target_properties(piper PROPERTIES INSTALL_RPATH "@executable_path/../lib")

# ---- Declare install targets ----

install(
  TARGETS piper
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

# Dependencies
install(
  DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  USE_SOURCE_PERMISSIONS  # keep +x
  FILES_MATCHING
  PATTERN "piper_phonemize"
  PATTERN "espeak-ng"
  PATTERN "*.dll"
)

install(
  DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  FILES_MATCHING
  PATTERN "*.dll"
  PATTERN "*.so*"
  PATTERN "*.dylib*"
)

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/data/espeak-ng-data
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(
  FILES ${CMAKE_CURRENT_LIST_DIR}/data/libtashkeel_model.ort
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

