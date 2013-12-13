#include <sstream>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <pcl_ros/transforms.h>

#include <pcl/search/kdtree.h>

#include "components/common.h"
#include "utilities/prettyprint.hpp"

std::string paramPath;
std::string outputPath;

ros::Publisher fpWorldPub;
ros::Publisher matchPub [5];

ros::Subscriber fpSub;
std::shared_ptr<tf::TransformListener> listener;

template <class P>
tf::Vector3 toVector3(P const& p) {
    tf::Vector3 res(p.x, p.y, p.z);
    return res;
}

void callback(sensor_msgs::PointCloud2ConstPtr msg) {
    // the feature points point cloud for previous scan
    static PointCloud::Ptr previousFPCloud;
    // the pose of the world frame in the previous sensor frame
    static tf::Transform previousTransform;

    // convertation
    PointCloud::Ptr fpCloud(new PointCloud);
    pcl::fromROSMsg(*msg, *fpCloud);
    ROS_INFO_STREAM("Received " << fpCloud->size() << " feature points in frame "
                                << msg->header.frame_id << " at " << msg->header.stamp);

    // look up for transform the receives sensor coordinates and gives world coordinates
    // that is from the old frame to the sensor frame
    tf::StampedTransform transform;
    try {
        listener->lookupTransform(
                // target frame
                "/world",
                // source frame
                msg->header.frame_id,
                ros::Time(0), transform);
    } catch (tf::TransformException ex) {
        ROS_ERROR_STREAM(ex.what());
        return;
    }

    // print the received transform
    geometry_msgs::Pose pose;
    tf::poseTFToMsg(transform, pose);
    ROS_INFO_STREAM("World pose in the sensor frame:\n" << pose);

    // get feature points in world coordinates
    PointCloud::Ptr fpTransformed(new PointCloud);
    pcl_ros::transformPointCloud(*fpCloud, *fpTransformed, transform);
    // publish the feature point cloud in the world frame
    {
        sensor_msgs::PointCloud2 resMsg;
        pcl::toROSMsg(*fpTransformed, resMsg);
        resMsg.header = msg->header;
        resMsg.header.frame_id = "/world";
        fpWorldPub.publish(resMsg);
    }

    // build kd-tree to search close feature points in world coordinates
    pcl::search::KdTree<PointType> kdtree;
    kdtree.setInputCloud(fpTransformed);

    // calculate the sensor shift
    float travelled = 0.0;
    {
        tf::Transform newSensorInOldSensorFrame
            = transform.inverse() * previousTransform;
        travelled = sqrt(newSensorInOldSensorFrame.getOrigin().length2());
        ROS_INFO_STREAM("moved on " << travelled);
    }

    // calculate how many of the feature points are stable (and to what extent)
    std::vector<int> survived(5);
    std::vector<PointCloud> matched(5);
    if (previousFPCloud) {
        // take all old points
        for (PointType point : previousFPCloud->points) {
            // search the closest in the world
            std::vector<int> idx;
            std::vector<float> dist2;
            kdtree.nearestKSearch(point, 1, idx, dist2);

            for (int i = 1; i <= 5; ++i) {
                bool ok = sqrt(dist2[0]) <= 0.01 * i;
                survived[i - 1] += ok;
                if (ok) {
                    matched[i - 1].push_back(point);
                }
            }
        }

        {
            std::ofstream ofstr(outputPath, std::ios_base::app);
            ofstr << survived << "\t" << travelled << '\n';
        }

        ROS_INFO_STREAM(survived << " feature points survived");
    }

    // publish really good matches
    {
        for (int i = 1; i <= 5; ++i) {
            sensor_msgs::PointCloud2 resMsg;
            pcl::toROSMsg(matched[i - 1], resMsg);
            resMsg.header = msg->header;
            resMsg.header.frame_id = "/world";
            matchPub[i - 1].publish(resMsg);
        }
    }

    // remember last feature point cloud
    previousFPCloud = fpTransformed;
    previousTransform = transform;
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "svs_alpha_node");

    ros::NodeHandle n("~");
    n.getParam("/svs/parampath", paramPath);
    n.getParam("outputpath", outputPath);
    ROS_INFO_STREAM("Parameters path is " << paramPath);
    ROS_INFO_STREAM("Output path is " << outputPath);

    {
        // clear output file
        std::ofstream ofstr(outputPath);
    }

    listener.reset(new tf::TransformListener);
    fpSub = n.subscribe("/svs/fp", 10, &callback);

    fpWorldPub = n.advertise<sensor_msgs::PointCloud2>("fp_world", 10);
    for (int i = 1; i <= 5; ++i) {
        std::stringstream sstr;
        sstr << "matched" << i;
        matchPub[i - 1] = n.advertise<sensor_msgs::PointCloud2>(sstr.str().c_str(), 10);
    }

    ros::spin();
    return 0;
}
