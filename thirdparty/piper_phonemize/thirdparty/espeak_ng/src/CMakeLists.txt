add_library(espeak-include INTERFACE)
target_include_directories(espeak-include INTERFACE include include/compat)

add_subdirectory(ucd-tools)
add_subdirectory(speechPlayer)
add_subdirectory(libespeak-ng)
