cmake_minimum_required(VERSION 2.8.3)
project( sptam )

# TODO check for SPTAM_LIBRARIES to be defined

add_executable( sptam_node main.cpp sptam_node.cpp )
target_link_libraries( sptam_node ${SPTAM_LIBRARIES} )
