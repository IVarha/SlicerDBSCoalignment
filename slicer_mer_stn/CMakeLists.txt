cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(subcortical_dbs)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/slicer_mer_stn")
set(EXTENSION_CATEGORY "DBS")
set(EXTENSION_CONTRIBUTORS "Igor Varga")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/slicer_mer_stn.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/slicer_mer_stn/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(load_nifty)
add_subdirectory(STNSegmenter)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
