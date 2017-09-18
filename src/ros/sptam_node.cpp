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

#include <iomanip>
#include "sptam_node.hpp"
#include "../sptam/sptam.hpp"
#include "../sptam/FeatureExtractorThread.hpp"
#include "../sptam/utils/projective_math.hpp"

#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/stereo_camera_model.h>
#include <image_geometry/pinhole_camera_model.h>

#ifdef SHOW_PROFILING

  #include "../sptam/utils/Profiler.hpp"
  #include "../sptam/utils/Logger.hpp"

#endif // SHOW_PROFILING

// Convert from CameraPose to tf::Pose (position and orientation)
inline void CameraPose2TFPose(const CameraPose& cameraPose, tf::Pose& pose)
{
  cv::Vec4d orientation = cameraPose.GetOrientationQuaternion();
  cv::Vec3d position = cameraPose.GetPosition();
  pose.setOrigin(tf::Vector3(position[0], position[1], position[2]));
  pose.setRotation(tf::Quaternion(orientation[1], orientation[2], orientation[3], orientation[0] )); // q = (x,y,z,w)
}

// Convert from tf::Pose to CameraPose (position and orientation)
inline void TFPose2CameraPose(const tf::Pose& pose, CameraPose& cameraPose)
{
  // convert to position opencv vector
  tf::Vector3 position_tf = pose.getOrigin();
  cv::Point3d position = cv::Point3d(position_tf.getX(), position_tf.getY(), position_tf.getZ());

  // Convert to orientation opencv quaternion
  tf::Quaternion orientation_tf = pose.getRotation();
  cv::Vec4d orientation(orientation_tf.getW(), orientation_tf.getX(), orientation_tf.getY(), orientation_tf.getZ());

  cameraPose = CameraPose(position, orientation);
}

// Set Opencv Algorithm parameters from ROS parameter server
void setParameters( ros::NodeHandle& nodeHandle, cv::Ptr<cv::Algorithm>&& algorithm, const std::string& base_name )
{
#if CV_MAJOR_VERSION == 2
  std::vector<cv::String> parameters;
  algorithm->getParams( parameters );

  for ( const auto& param : parameters )
  {
    if ( nodeHandle.hasParam(base_name + "/" + param) )
    {
      int param_type = algorithm->paramType( param );

      switch ( param_type )
      {
        case cv::Param::INT:
        {
          int val;
          nodeHandle.getParam(base_name + "/" + param, val);
          algorithm->set(param, val);
          std::cout << "  " << param << ": " << val << std::endl;
          break;
        }

        case cv::Param::BOOLEAN:
        {
          bool val;
          nodeHandle.getParam(base_name + "/" + param, val);
          algorithm->set(param, val);
          std::cout << "  " << param << ": " << val << std::endl;
          break;
        }

        case cv::Param::REAL:
        {
          double val;
          nodeHandle.getParam(base_name + "/" + param, val);
          algorithm->set(param, val);
          std::cout << "  " << param << ": " << val << std::endl;
          break;
        }

        case cv::Param::STRING:
        {
          std::string val;
          nodeHandle.getParam(base_name + "/" + param, val);
          algorithm->set(param, val);
          std::cout << "  " << param << ": " << val << std::endl;
          break;
        }

        default:
          ROS_ERROR_STREAM("unknown/unsupported parameter type for ");
          break;
      }
    }
  }
#else

  //ds ugly dugly - hardcode all available opencv parameters so we can switch bomb it to the virtual methods
  std::vector<std::string> parameters(4);
  parameters[0] = "nfeatures";
  parameters[1] = "minDistance";
  parameters[2] = "qualityLevel";
  parameters[3] = "useHarrisDetector";
//  parameters[4] = "bytes";
//  parameters[5] = "crossCheck";

  //ds currently only the GFTT detector configuration is enabled
  cv::Ptr<cv::GFTTDetector> detector = algorithm.dynamicCast<cv::GFTTDetector>();

  //ds if the cast succeeded
  if (detector) {

    //ds for all feasible parameters
    for (const std::string& parameter: parameters) {

      //ds if the parameter is available in the ROS node
      const std::string ros_parameter = base_name + "/" + parameter;
      if (nodeHandle.hasParam(ros_parameter)) {

        //ds call corresponding method
        if (parameter == "nfeatures") {
          int32_t value;
          nodeHandle.getParam(ros_parameter, value);
          std::cerr << " " << parameter << ": " << value << std::endl;
          detector->setMaxFeatures(value);
        } else if (parameter == "minDistance") {
          double value;
          nodeHandle.getParam(ros_parameter, value);
          std::cerr << " " << parameter << ": " << value << std::endl;
          detector->setMinDistance(value);
        } else if (parameter == "qualityLevel") {
          double value;
          nodeHandle.getParam(ros_parameter, value);
          std::cerr << " " << parameter << ": " << value << std::endl;
          detector->setQualityLevel(value);
        } else if (parameter == "useHarrisDetector") {
          bool value;
          nodeHandle.getParam(ros_parameter, value);
          std::cerr << " " << parameter << ": " << value << std::endl;
          detector->setHarrisDetector(value);
        } else {
          ROS_ERROR_STREAM("unknown/unsupported parameter type for");
        }
      }
    }
  }
#endif
}

// ================================================================== //

namespace sptam {

sptam_node::sptam_node(ros::NodeHandle& nh, ros::NodeHandle& nhp)
  :
  sptam_(nullptr),
  motionModel_(new MotionModel(cv::Point3d(0,0,0), cv::Vec4d(1,0,0,0))),
  it_( nhp ),
  odom_to_map_(tf::Transform::getIdentity()),
  transform_thread_(nullptr)
{
  // Get node parameters
  nhp.param<std::string>("odom_frame", odom_frame_, "/odom");
  nhp.param<std::string>("base_link_frame", base_frame_, "/base_link");
  nhp.param<std::string>("camera_frame", camera_frame_, "/camera");
  nhp.param<std::string>("map_frame", map_frame_, "/map");
  nhp.param<bool>("use_odometry", use_odometry_, false);
  nhp.param<double>("transform_publish_freq", transform_publish_freq_, 30.0);
  nhp.param<double>("tf_delay", tf_delay_, 1.0/transform_publish_freq_);

  bool use_approx_sync;
  nhp.param<bool>("approximate_sync", use_approx_sync, false);

  // load feature detector
  {
    std::string detectorName;
    nhp.param<std::string>("FeatureDetector/Name", detectorName, "GFTT");

    std::cout << "detector: " << detectorName << std::endl;
#if CV_MAJOR_VERSION == 2
    featureDetector_ = cv::FeatureDetector::create(detectorName);
#else
    featureDetector_ = cv::GFTTDetector::create();
#endif

    if ( not featureDetector_ )
      ROS_ERROR_STREAM("could not load feature detector with name " << detectorName);

    setParameters( nhp, featureDetector_, "FeatureDetector" );
  }

  // load descriptor extractor
  {
    std::string extractorName;
    nhp.param<std::string>("DescriptorExtractor/Name", extractorName, "BRIEF");

    std::cout << "extractor: " << extractorName << std::endl;
#if CV_MAJOR_VERSION == 2
    descriptorExtractor_ = cv::DescriptorExtractor::create(extractorName);
#else
    descriptorExtractor_ = cv::xfeatures2d::BriefDescriptorExtractor::create();
#endif

    if ( not descriptorExtractor_ )
      ROS_ERROR_STREAM("could not load descriptor extractor with name " << extractorName);

    setParameters( nhp, descriptorExtractor_, "DescriptorExtractor" );
  }

  // load descriptor matcher
  {
    std::string matcherName;
    nhp.param<std::string>("DescriptorMatcher/Name", matcherName, "BruteForce-Hamming");

    std::cout << "matcher: " << matcherName << std::endl;
    mapper_params_.descriptorMatcher = cv::DescriptorMatcher::create( matcherName );

    if ( not mapper_params_.descriptorMatcher )
      ROS_ERROR_STREAM("could not load descriptor matcher with name " << matcherName);

    setParameters( nhp, mapper_params_.descriptorMatcher, "DescriptorMatcher" );
  }

  // Mapper Parameters
  // nhp.param is not overloaded for unsigned int
  int matchingCellSizeParam, framesBetweenKeyFramesParam, matchingNeighborhoodParam;
  nhp.param<int>("MatchingCellSize", matchingCellSizeParam, 15);
  mapper_params_.matchingCellSize = matchingCellSizeParam;
  nhp.param<double>("MatchingDistance", mapper_params_.matchingDistanceThreshold, 25.0);
  nhp.param<int>("MatchingNeighborhood", matchingNeighborhoodParam, 1);
  mapper_params_.matchingNeighborhoodThreshold = matchingNeighborhoodParam;
  nhp.param<double>("EpipolarDistance", mapper_params_.epipolarDistanceThreshold, 0.0);
  nhp.param<double>("KeyFrameDistance", mapper_params_.keyFrameDistanceThreshold, 0.0);
  nhp.param<int>("FramesBetweenKeyFrames", framesBetweenKeyFramesParam, 0);
  mapper_params_.framesBetweenKeyFrames = framesBetweenKeyFramesParam;

  // Camera Calibration Parameters
  nhp.param<double>("FrustumNearPlaneDist", cameraParametersLeft_.frustumNearPlaneDist, 0.1);
  nhp.param<double>("FrustumFarPlaneDist", cameraParametersLeft_.frustumFarPlaneDist, 100.0);
  cameraParametersRight_.frustumNearPlaneDist = cameraParametersLeft_.frustumNearPlaneDist;
  cameraParametersRight_.frustumFarPlaneDist = cameraParametersLeft_.frustumFarPlaneDist;

  // Create RowMatcher instance
  rowMatcher_ = new RowMatcher( mapper_params_.matchingDistanceThreshold, mapper_params_.descriptorMatcher, mapper_params_.epipolarDistanceThreshold );

  // Subscribe to images messages
#ifdef USE_EUROC_CALIBRATION
  sub_l_image_.subscribe(nhp, "/stereo/left/image_rect", 1);
  sub_r_image_.subscribe(nhp, "/stereo/right/image_rect", 1);

  if ( use_approx_sync )
  {
    approximate_sync_.reset( new ApproximateSync( ApproximatePolicy(10),
                                                  sub_l_image_,
                                                  sub_r_image_ ) );

    approximate_sync_->registerCallback( boost::bind( &sptam::sptam_node::onImages,
                                                      this, _1, _2 ) );
  }
  else
  {
    exact_sync_.reset( new ExactSync( ExactPolicy(1),
                                      sub_l_image_,
                                      sub_r_image_ ) );

    exact_sync_->registerCallback( boost::bind( &sptam::sptam_node::onImages,
                                                this, _1, _2 ) );
  }
#else
  sub_l_image_.subscribe(nhp, "/stereo/left/image_rect", 1);
  sub_l_info_ .subscribe(nhp, "/stereo/left/camera_info", 1);
  sub_r_image_.subscribe(nhp, "/stereo/right/image_rect", 1);
  sub_r_info_ .subscribe(nhp, "/stereo/right/camera_info", 1);

  if ( use_approx_sync )
  {
    approximate_sync_.reset( new ApproximateSync( ApproximatePolicy(10),
                                                  sub_l_image_, sub_l_info_,
                                                  sub_r_image_, sub_r_info_ ) );

    approximate_sync_->registerCallback( boost::bind( &sptam::sptam_node::onImages,
                                                      this, _1, _2, _3, _4 ) );
  }
  else
  {
    exact_sync_.reset( new ExactSync( ExactPolicy(1),
                                      sub_l_image_, sub_l_info_,
                                      sub_r_image_, sub_r_info_ ) );

    exact_sync_->registerCallback( boost::bind( &sptam::sptam_node::onImages,
                                                this, _1, _2, _3, _4 ) );
  }
#endif

  mapPub_ = nhp.advertise<sensor_msgs::PointCloud2>("point_cloud", 100);
  posePub_ = nhp.advertise<geometry_msgs::PoseStamped>("robot/pose", 100);

  leftKpPub_ = it_.advertise("/stereo/left/keypoints", 1);
  rightKpPub_ = it_.advertise("/stereo/right/keypoints", 1);

  // start map->odom transform publisher thread
  if ( use_odometry_ )
    // this loop periodically publishes map->odom transform from another thread
    transform_thread_ = new std::thread( boost::bind(&sptam::sptam_node::publishTransformLoop, this) );
  else
    // this loop periodically publishes map->base_link transform from another thread
    transform_thread_ = new std::thread( boost::bind(&sptam::sptam_node::publishTransformLoop2, this) );

  ROS_INFO_STREAM("sptam node initialized");
}

sptam_node::~sptam_node()
{
  if (_number_of_frames_processed > 0) {
    std::cerr << "sptam_node::~sptam_node|average processing time per frame (s): " << _processing_time_seconds/_number_of_frames_processed << std::endl;
  }

  ROS_INFO_STREAM("starting sptam node cleanup...");

  // transform_thread_ is null if odometry is disabled
  if ( transform_thread_ ) {
    ROS_INFO_STREAM("wait for transform publisher thread to join...");
    transform_thread_->join();
    delete transform_thread_;
  }

  ROS_INFO_STREAM("stopping sptam threads...");
  sptam_->stop();

  #ifdef SHOW_PROFILING
    for ( const auto& mapPoint : map_.GetMapPoints() ) {
      WriteToLog( " tk MeasurementCount: ", mapPoint->GetMeasurementCount() );
    }
  #endif

  #ifdef SHOW_PROFILING
    for ( const auto& keyFrame : map_.GetKeyFrames() ) {
      CameraPose keyFramePose = keyFrame->GetCameraPose();
      WriteToLog("BASE_LINK_KF:", keyFrame->GetId(), keyFramePose.GetPosition(), keyFramePose.GetOrientationMatrix());
    }
  #endif

  // create map file
  {
    std::ofstream out("map cloud.dat");
    for ( const auto& point : map_.GetMapPoints() )
      out << point->GetPosition().x << " " << point->GetPosition().y <<  " " << point->GetPosition().z << std::endl;
    out.close();
  }

  ROS_INFO_STREAM("done!");

  // sleep for one second
  ros::Duration( 1.0 ).sleep();
}

bool sptam_node::getCameraInOdom(tf::StampedTransform& camera_to_odom, const ros::Time& t)
{
  // lookupTransform(target_frame, source_frame ...)
  if ( transform_listener_.waitForTransform(odom_frame_, camera_frame_, t, ros::Duration(0.1)) ) {
    transform_listener_.lookupTransform(odom_frame_, camera_frame_, t, camera_to_odom);
  }
  else {
    ROS_WARN("Failed to retrieve camera pose in odom frame");
    return false;
  }

  return true;
}

bool sptam_node::getBaseLinkPose(const CameraPose& cameraPose, const ros::Time& t, tf::Pose& base_to_map)
{
  tf::StampedTransform base_to_camera;

  // lookupTransform(target_frame, source_frame ...)
  if ( transform_listener_.waitForTransform(camera_frame_, base_frame_, t, ros::Duration(0.1)) ) {
    transform_listener_.lookupTransform(camera_frame_, base_frame_, t, base_to_camera);
  }
  else {
    ROS_WARN("Failed to retrieve camera pose in odom frame");
    return false;
  }

  tf::Pose camera_to_map;
  CameraPose2TFPose( cameraPose, camera_to_map );

  base_to_map = camera_to_map * base_to_camera;

  return true;
}

void sptam_node::fixOdomFrame(const CameraPose& cameraPose, const tf::StampedTransform& camera_to_odom, const ros::Time& t)
{
  tf::Pose camera_to_map;
  CameraPose2TFPose( cameraPose, camera_to_map );

  // compute the new difference between map and odom
  tf::Transform odom_to_map = camera_to_map * camera_to_odom.inverse();

  std::lock_guard<std::mutex> lock( odom_to_map_mutex_ );
  odom_to_map_ = odom_to_map;
}

#ifdef USE_EUROC_CALIBRATION
void sptam_node::onImages(const sensor_msgs::ImageConstPtr& l_image_msg, const sensor_msgs::ImageConstPtr& r_image_msg)
#else
void sptam::sptam_node::onImages(
  const sensor_msgs::ImageConstPtr& l_image_msg, const sensor_msgs::CameraInfoConstPtr& left_info,
  const sensor_msgs::ImageConstPtr& r_image_msg, const sensor_msgs::CameraInfoConstPtr& right_info
)
#endif
{
  //ds processing start
  std::chrono::time_point<std::chrono::system_clock> time_begin = std::chrono::system_clock::now();

  ROS_INFO_STREAM("dt: " << (l_image_msg->header.stamp - r_image_msg->header.stamp).toSec());

  // Save current Time
  ros::Time currentTime = l_image_msg->header.stamp;

  // If using odometry, try to get a new estimate for the current pose.
  // if not using odometry, camera_to_odom is left blank.
  tf::StampedTransform camera_to_odom;
  if ( use_odometry_ )
  {
    if ( not getCameraInOdom(camera_to_odom, currentTime) )
      return;

    TFPose2CameraPose(odom_to_map_ * camera_to_odom, cameraPose_);
  }
  else
  {
    cv::Point3d currentCameraPosition;
    cv::Vec4d currentCameraOrientation;
    motionModel_->PredictNextCameraPose(currentCameraPosition, currentCameraOrientation);
    cameraPose_ = CameraPose( currentCameraPosition, currentCameraOrientation );
  }

  // If SPTAM has not been initialized yet, do it
  if ( sptam_ == nullptr )
  {
    ROS_INFO_STREAM("init calib");
#ifdef USE_EUROC_CALIBRATION
    loadCameraCalibration(0, 0);
#else
    loadCameraCalibration(left_info, right_info);
#endif
  }

  // convert image to OpenCv cv::Mat format
  cv_bridge::CvImageConstPtr bridgeLeft_ptr = cv_bridge::toCvShare(l_image_msg, "rgb8");
  cv_bridge::CvImageConstPtr bridgeRight_ptr = cv_bridge::toCvShare(r_image_msg, "rgb8");

  // save images
  cv::Mat imageLeft = bridgeLeft_ptr->image;
  cv::Mat imageRight = bridgeRight_ptr->image;

  //ds rectify image (euroc)
#ifdef USE_EUROC_CALIBRATION
  cv::remap(imageLeft, imageLeft, _undistort_rectify_maps_left[0], _undistort_rectify_maps_left[1], cv::INTER_LINEAR);
  cv::remap(imageRight, imageRight, _undistort_rectify_maps_right[0], _undistort_rectify_maps_right[1], cv::INTER_LINEAR);
#endif

  #ifdef SHOW_PROFILING
      double start, end;
      start = GetSeg();
  #endif // SHOW_PROFILING

  #ifdef SHOW_PROFILING
    double startStep, endStep;
    startStep = GetSeg();
  #endif

  FeatureExtractorThread featureExtractorThreadLeft(imageLeft, featureDetector_, descriptorExtractor_);
  FeatureExtractorThread featureExtractorThreadRight(imageRight, featureDetector_, descriptorExtractor_);

  featureExtractorThreadLeft.WaitUntilFinished();
  const std::vector<cv::KeyPoint>& keyPointsLeft = featureExtractorThreadLeft.GetKeyPoints();
  const cv::Mat& descriptorsLeft = featureExtractorThreadLeft.GetDescriptors();

  featureExtractorThreadRight.WaitUntilFinished();
  const std::vector<cv::KeyPoint>& keyPointsRight = featureExtractorThreadRight.GetKeyPoints();
  const cv::Mat& descriptorsRight = featureExtractorThreadRight.GetDescriptors();

  ImageFeatures imageFeaturesLeft = ImageFeatures(imageLeft, keyPointsLeft, descriptorsLeft, mapper_params_.matchingCellSize);
  ImageFeatures imageFeaturesRight = ImageFeatures(imageRight, keyPointsRight, descriptorsRight, mapper_params_.matchingCellSize);

  #ifdef SHOW_PROFILING
    endStep = GetSeg();
    WriteToLog(" tk Extract: ", startStep, endStep);
  #endif

  // if the map was not initialized, try to build it
  if( not map_.nMapPoints() )
  {
    ROS_INFO("Trying to intialize map...");

    // Create keyFrame
    StereoFrame* frame = new StereoFrame(
      cameraPose_, cameraParametersLeft_,
      stereo_baseline_, cameraParametersRight_,
      imageFeaturesLeft, imageFeaturesRight, true
    );

    frame->SetId( 0 ); // Set Keyframe ID

    // Initialize MapMaker
    // TODO: this bool must be a local variable to do not check not map_.nMapPoints() in the "if"
    /*bool isMapInitialized = */InitFromStereo( map_, *frame, imageLeft, *rowMatcher_);
  }
  // if the map is already initialized, do tracking
  else
  {

    cameraPose_ = sptam_->track(cameraPose_, imageFeaturesLeft, imageFeaturesRight, imageLeft, imageRight);

    #ifdef SHOW_PROFILING
      end = GetSeg();
      std::cout << GetSeg() << " tk trackingtotal: " << (end - start) << std::endl;

      std::stringstream message;
      message << std::fixed <<  GetSeg() << " tk trackingtotal: " << (end - start) << std::endl;
      Logger::Write( message.str() );
    #endif

    // fix the error between map and odom to correct the current pose.
    if ( use_odometry_ )
      fixOdomFrame( cameraPose_, camera_to_odom, currentTime );
    else
    // if odometry is disabled, odom_to_map_ actually contains the map->cam transform because I'm too lazy
    {
      motionModel_->UpdateCameraPose(cameraPose_.GetPosition(), cameraPose_.GetOrientationQuaternion());

      tf::Transform cam_to_map;
      CameraPose2TFPose(cameraPose_, cam_to_map);

      tf::StampedTransform base_to_cam;
      transform_listener_.lookupTransform(camera_frame_, base_frame_, currentTime, base_to_cam);

      std::lock_guard<std::mutex> lock( odom_to_map_mutex_ );
      odom_to_map_ = cam_to_map * base_to_cam;
    }
  }

  //ds processing stop
  _processing_time_seconds += static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now()-time_begin).count();
  ++_number_of_frames_processed;

  //ds save current pose
  _writePoseToFile(l_image_msg);

  // Publish Map To be drawn by rviz visualizer
  publishMap();

  // Publish the camera Pose
  publishPose( l_image_msg->header.seq, currentTime, cameraPose_ );
}

void sptam_node::_writePoseToFile(const sensor_msgs::ImageConstPtr& image_left_) const {

  //ds construct eigen camera pose
  const cv::Matx33d orientation = cameraPose_.GetOrientationMatrix();
  const cv::Vec3d position      = cameraPose_.GetPosition();
  Eigen::Isometry3d camera_to_world(Eigen::Isometry3d::Identity());
  camera_to_world.linear() << orientation.val[0], orientation.val[1], orientation.val[2],
                              orientation.val[3], orientation.val[4], orientation.val[5],
                              orientation.val[6], orientation.val[7], orientation.val[8];
  camera_to_world.translation() << position.val[0], position.val[1], position.val[2];

  //ds file streams
  std::ofstream outfile_trajectory_kitti;
  std::ofstream outfile_trajectory_tum;

  //ds open file streams - overwritting the first and afterwards otherwise
  if (_number_of_frames_processed == 1) {
    outfile_trajectory_kitti.open(_working_directory + "sptam_trajectory_kitti.txt", std::ifstream::out);
    outfile_trajectory_tum.open(_working_directory + "sptam_trajectory_tum.txt", std::ifstream::out);
  } else {
    outfile_trajectory_kitti.open(_working_directory + "sptam_trajectory_kitti.txt", std::ifstream::app);
    outfile_trajectory_tum.open(_working_directory + "sptam_trajectory_tum.txt", std::ifstream::app);
  }
  outfile_trajectory_kitti << std::fixed;
  outfile_trajectory_tum << std::fixed;
  outfile_trajectory_kitti << std::setprecision(9);
  outfile_trajectory_tum << std::setprecision(9);

  //ds save timestamp for tum
  const double timestamp_seconds = image_left_->header.stamp.sec+image_left_->header.stamp.nsec/1e9;
  outfile_trajectory_tum << timestamp_seconds << " ";

  //ds dump transform according to KITTI format
  for (uint8_t u = 0; u < 3; ++u) {
    for (uint8_t v = 0; v < 4; ++v) {
      outfile_trajectory_tum << camera_to_world(u,v) << " ";
      outfile_trajectory_kitti << camera_to_world(u,v) << " ";
    }
  }
  outfile_trajectory_kitti << "\n";
  outfile_trajectory_tum << "\n";
  outfile_trajectory_kitti.close();
  outfile_trajectory_tum.close();
}

void sptam_node::loadCameraCalibration(const sensor_msgs::CameraInfoConstPtr& left_info, const sensor_msgs::CameraInfoConstPtr& right_info) {

  //ds brutal - use hardcoded euroc camera calibration
#ifdef USE_EUROC_CALIBRATION
  cv::Size image_size(752, 480);
  cv::Mat camera_calibration_matrix_left(cv::Mat::eye(3, 3, CV_64F));
  cv::Mat camera_calibration_matrix_right(cv::Mat::eye(3, 3, CV_64F));
  cv::Mat distortion_coefficients_left(cv::Mat::zeros(4, 1, CV_64F));
  cv::Mat distortion_coefficients_right(cv::Mat::zeros(4, 1, CV_64F));
  cv::Mat projection_matrix_left(cv::Mat::eye(3, 4, CV_64F));
  cv::Mat projection_matrix_right(cv::Mat::eye(3, 4, CV_64F));
  cv::Mat rectification_matrix_left(cv::Mat::eye(3, 3, CV_64F));
  cv::Mat rectification_matrix_right(cv::Mat::eye(3, 3, CV_64F));

  camera_calibration_matrix_left.at<double>(0,0)  = 458.654;
  camera_calibration_matrix_left.at<double>(0,2)  = 367.215;
  camera_calibration_matrix_left.at<double>(1,1)  = 457.296;
  camera_calibration_matrix_left.at<double>(1,2)  = 248.375;
  camera_calibration_matrix_right.at<double>(0,0) = 457.587;
  camera_calibration_matrix_right.at<double>(0,2) = 379.999;
  camera_calibration_matrix_right.at<double>(1,1) = 456.134;
  camera_calibration_matrix_right.at<double>(1,2) = 255.238;

  distortion_coefficients_left.at<double>(0)  = -0.28340811;
  distortion_coefficients_left.at<double>(1)  = 0.07395907;
  distortion_coefficients_left.at<double>(2)  = 0.00019359;
  distortion_coefficients_left.at<double>(3)  = 1.76187114e-05;
  distortion_coefficients_right.at<double>(0) = -0.28368365;
  distortion_coefficients_right.at<double>(1) = 0.07451284;
  distortion_coefficients_right.at<double>(2) = -0.00010473;
  distortion_coefficients_right.at<double>(3) = -3.55590700e-05;

  rectification_matrix_left.at<double>(0,0) = 0.999966347530033;
  rectification_matrix_left.at<double>(0,1) = -0.001422739138722922;
  rectification_matrix_left.at<double>(0,2) = 0.008079580483432283;
  rectification_matrix_left.at<double>(1,0) = 0.001365741834644127;
  rectification_matrix_left.at<double>(1,1) = 0.9999741760894847;
  rectification_matrix_left.at<double>(1,2) = 0.007055629199258132;
  rectification_matrix_left.at<double>(2,0) = -0.008089410156878961;
  rectification_matrix_left.at<double>(2,1) = -0.007044357138835809;
  rectification_matrix_left.at<double>(2,2) = 0.9999424675829176;

  rectification_matrix_right.at<double>(0,0) = 0.9999633526194376;
  rectification_matrix_right.at<double>(0,1) = -0.003625811871560086;
  rectification_matrix_right.at<double>(0,2) = 0.007755443660172947;
  rectification_matrix_right.at<double>(1,0) = 0.003680398547259526;
  rectification_matrix_right.at<double>(1,1) = 0.9999684752771629;
  rectification_matrix_right.at<double>(1,2) = -0.007035845251224894;
  rectification_matrix_right.at<double>(2,0) = -0.007729688520722713;
  rectification_matrix_right.at<double>(2,1) = 0.007064130529506649;
  rectification_matrix_right.at<double>(2,2) = 0.999945173484644;

  projection_matrix_left.at<double>(0,0) = 435.2046959714599;
  projection_matrix_left.at<double>(0,2) = 367.4517211914062;
  projection_matrix_left.at<double>(1,1) = 435.2046959714599;
  projection_matrix_left.at<double>(1,2) = 252.2008514404297;

  projection_matrix_right.at<double>(0,0) = 435.2046959714599;
  projection_matrix_right.at<double>(0,2) = 367.4517211914062;
  projection_matrix_right.at<double>(0,3) = -47.90639384423901;
  projection_matrix_right.at<double>(1,1) = 435.2046959714599;
  projection_matrix_right.at<double>(1,2) = 252.2008514404297;

  //ds compute undistorted and rectified mappings
  cv::initUndistortRectifyMap(camera_calibration_matrix_left,
                              distortion_coefficients_left,
                              rectification_matrix_left,
                              projection_matrix_left,
                              image_size,
                              CV_16SC2,
                              _undistort_rectify_maps_left[0],
                              _undistort_rectify_maps_left[1]);
  cv::initUndistortRectifyMap(camera_calibration_matrix_right,
                              distortion_coefficients_right,
                              rectification_matrix_right,
                              projection_matrix_right,
                              image_size,
                              CV_16SC2,
                              _undistort_rectify_maps_right[0],
                              _undistort_rectify_maps_right[1]);

  //ds compute baseline
  const double baseline = -projection_matrix_right.at<double>(0,3) / projection_matrix_right.at<double>(0,0);

  // Ponemos que el frame id de las camara info sea el mismo ???
  ROS_INFO_STREAM("tx: " << projection_matrix_right.at<double>(0,3)
              << " fx: " << projection_matrix_right.at<double>(0,0)
              << " baseline: " << baseline);

  // Get rectify intrinsic Matrix (is the same for both cameras because they are rectify)
  cv::Matx33d intrinsic = projection_matrix_left( cv::Rect(0,0,3,3) );

  // Save rectify intrinsic Matrix
  cameraParametersLeft_.intrinsic = intrinsic;
  cameraParametersRight_.intrinsic = intrinsic;

  // Save the baseline
  stereo_baseline_ = baseline;

  // Compute Fild Of View (Frustum)

  cameraParametersLeft_.horizontalFOV = computeFOV( intrinsic(0, 0), image_size.width );
  cameraParametersLeft_.verticalFOV = computeFOV( intrinsic(1, 1), image_size.height );

  cameraParametersRight_.horizontalFOV = computeFOV( intrinsic(0, 0), image_size.width );
  cameraParametersRight_.verticalFOV = computeFOV( intrinsic(1, 1), image_size.height );

  // Check if a valid calibration exists
#else
  if (left_info->K[0] == 0.0) {
    ROS_ERROR("La camara no esta calibrada");
    return;
  }

  // Ponemos que el frame id de las camara info sea el mismo
  sensor_msgs::CameraInfoPtr left_info_copy = boost::make_shared<sensor_msgs::CameraInfo>(*left_info);
  sensor_msgs::CameraInfoPtr right_info_copy = boost::make_shared<sensor_msgs::CameraInfo>(*right_info);
  left_info_copy->header.frame_id = "stereo";
  right_info_copy->header.frame_id = "stereo";

  ROS_INFO_STREAM("tx: " << right_info->P[3] << " fx: " << right_info->K[0] << " baseline: " << right_info->P[3] / right_info->K[0]);

  // Get Stereo Camera Model from Camera Info message
  image_geometry::StereoCameraModel stereoCameraModel;
  stereoCameraModel.fromCameraInfo(left_info_copy, right_info_copy);

  // Get PinHole Camera Model from the Stereo Camera Model
  const image_geometry::PinholeCameraModel& cameraLeft = stereoCameraModel.left();

  // Get rectify intrinsic Matrix (is the same for both cameras because they are rectify)
  cv::Mat projection = cv::Mat( cameraLeft.projectionMatrix() );
  cv::Matx33d intrinsic = projection( cv::Rect(0,0,3,3) );

  // Save rectify intrinsic Matrix
  cameraParametersLeft_.intrinsic = intrinsic;
  cameraParametersRight_.intrinsic = intrinsic;

  // Save the baseline
  stereo_baseline_ = stereoCameraModel.baseline();

  // Compute Fild Of View (Frustum)

  cameraParametersLeft_.horizontalFOV = computeFOV( intrinsic(0, 0), left_info_copy->width );
  cameraParametersLeft_.verticalFOV = computeFOV( intrinsic(1, 1), left_info_copy->height );

  cameraParametersRight_.horizontalFOV = computeFOV( intrinsic(0, 0), right_info_copy->width );
  cameraParametersRight_.verticalFOV = computeFOV( intrinsic(1, 1), right_info_copy->height );
#endif

  ROS_INFO_STREAM("baseline: " << stereo_baseline_);

  // Create SPTAM instance
  sptam_ = new SPTAM(
    map_,
    cameraParametersLeft_, cameraParametersRight_, stereo_baseline_,
    *rowMatcher_, mapper_params_
  );
}

void sptam_node::publishMap()
{
  if ( mapPub_.getNumSubscribers() < 1 )
    return;

  // Create PointCloud message for visualization
  pcl::PointCloud<pcl::PointXYZRGB> msg;
  msg.header.frame_id = map_frame_;
  msg.height = 1;
  msg.width = map_.nMapPoints();
  for(const auto& mapPoint : map_.GetMapPoints() )
  {
    // Get Point from Map
    cv::Point3d point3d = mapPoint->GetPosition();
    pcl::PointXYZRGB point_pcl;
    point_pcl.x = point3d.x;
    point_pcl.y = point3d.y;
    point_pcl.z = point3d.z;
    point_pcl.r = mapPoint->color(0);
    point_pcl.g = mapPoint->color(1);
    point_pcl.b = mapPoint->color(2);

    msg.points.push_back ( point_pcl );
  }

  // Publish the PointCloud
  mapPub_.publish( msg );
}

void sptam_node::publishPose(const uint32_t seq, const ros::Time& time, const CameraPose& currentCameraPose)
{
  tf::Pose base_to_map;
  if ( not getBaseLinkPose( currentCameraPose, time, base_to_map ) )
    return;

  #ifdef SHOW_PROFILING
    const tf::Vector3& position = base_to_map.getOrigin();
    const tf::Matrix3x3& orientation = base_to_map.getBasis();

    std::stringstream message;

    message << std::fixed << "BASE_LINK_POSE:" << " " << seq << " " << time.toSec() << " "
      << orientation[0][0] << " " << orientation[0][1] << " " << orientation[0][2] << " " << position.x() << " "
      << orientation[1][0] << " " << orientation[1][1] << " " << orientation[1][2] << " " << position.y() << " "
      << orientation[2][0] << " " << orientation[2][1] << " " << orientation[2][2] << " " << position.z() << " "
      << std::endl;

    Logger::Write( message.str() );
  #endif

  geometry_msgs::PoseStamped msg;

  msg.header.seq = seq;
  msg.header.stamp = time;
  msg.header.frame_id = map_frame_;

  msg.pose.orientation.x = base_to_map.getRotation().x();
  msg.pose.orientation.y = base_to_map.getRotation().y();
  msg.pose.orientation.z = base_to_map.getRotation().z();
  msg.pose.orientation.w = base_to_map.getRotation().w();

  msg.pose.position.x = base_to_map.getOrigin().x();
  msg.pose.position.y = base_to_map.getOrigin().y();
  msg.pose.position.z = base_to_map.getOrigin().z();

  // Publish the camera pose
  posePub_.publish( msg );
}

void sptam_node::publishTransform()
{
  std::lock_guard<std::mutex> lock( odom_to_map_mutex_ );

  // TODO sacar el delay de acá y los parámetros si estamos seguros que no hace falta
  ros::Time tf_expiration = ros::Time::now()/* + ros::Duration( tf_delay_ )*/;

  transform_broadcaster_.sendTransform(tf::StampedTransform(odom_to_map_, tf_expiration, map_frame_, odom_frame_));
}

void sptam_node::publishTransformLoop()
{
  if ( transform_publish_freq_ == 0 )
    return;

  ros::Rate r( transform_publish_freq_ );

  while ( ros::ok() ) {
    publishTransform();
    r.sleep();
  }
}

void sptam_node::publishTransform2()
{
  std::lock_guard<std::mutex> lock( odom_to_map_mutex_ );

  // TODO: remove the delay and parameters from here if we are sure that they are not needed
  ros::Time tf_expiration = ros::Time::now()/* + ros::Duration( tf_delay_ )*/;

  // if odometry is disabled, odom_to_map_ actually contains the map->base transform because I'm too lazy
  transform_broadcaster_.sendTransform(tf::StampedTransform(odom_to_map_, tf_expiration, map_frame_, base_frame_));
}

void sptam_node::publishTransformLoop2()
{
  if ( transform_publish_freq_ == 0 )
    return;

  ros::Rate r( transform_publish_freq_ );

  while ( ros::ok() ) {
    publishTransform2();
    r.sleep();
  }
}
}
