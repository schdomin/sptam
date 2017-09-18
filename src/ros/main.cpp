/**
 * This file is part of S-PTAM.
 *
 * Copyright (C) 2015 Taihú Pire and Thomas Fischer
 * For more information see <https://github.com/lrse/sptam>
 *
 * S-PTAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * S-PTAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with S-PTAM. If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors:  Taihú Pire <tpire at dc dot uba dot ar>
 *           Thomas Fischer <tfischer at dc dot uba dot ar>
 *
 * Laboratory of Robotics and Embedded Systems
 * Department of Computer Science
 * Faculty of Exact and Natural Sciences
 * University of Buenos Aires
 */

#include <ros/ros.h>
#include "sptam_node.hpp"

int main(int argc, char *argv[])
{
  // Override SIGINT handler
  ros::init(argc, argv, "S_PTAM");

  ROS_INFO("S-PTAM node running...");

  // Stereo slam class
  ros::NodeHandle nodeHandle;
  ros::NodeHandle nodeHandle_private("~");

  // Create sptam instance
  sptam::sptam_node sptam_node(nodeHandle, nodeHandle_private);

  //ds set up working directory for trajectory saving
  std::string working_directory(argv[0]);
  const std::string::size_type index_separator(working_directory.find_last_of("/"));
  working_directory = working_directory.substr(0, index_separator+1);
  sptam_node.setWorkingDirectory(working_directory);
  ROS_INFO("saving trajectories to: %s", working_directory.c_str());

  ros::spin();

  return 0;
}
