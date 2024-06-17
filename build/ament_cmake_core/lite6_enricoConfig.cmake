# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_lite6_enrico_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED lite6_enrico_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(lite6_enrico_FOUND FALSE)
  elseif(NOT lite6_enrico_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(lite6_enrico_FOUND FALSE)
  endif()
  return()
endif()
set(_lite6_enrico_CONFIG_INCLUDED TRUE)

# output package information
if(NOT lite6_enrico_FIND_QUIETLY)
  message(STATUS "Found lite6_enrico: 0.3.0 (${lite6_enrico_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'lite6_enrico' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${lite6_enrico_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(lite6_enrico_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${lite6_enrico_DIR}/${_extra}")
endforeach()
