#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <svs/Alphas.h>
#include "components/svs.h"

ros::Publisher alphaPublisher;
ros::Subscriber cloudSubscriber;

int cutWidth = 120;
int cutHeight = 90;

void callback(sensor_msgs::PointCloud2ConstPtr msg) {
    ros::WallTime before = ros::WallTime::now();

    PointCloud cloud;
    pcl::fromROSMsg(*msg, cloud);
    ROS_INFO_STREAM("Received and converted " << cloud.width << "x" << cloud.height << " cloud");

    PointCloud::Ptr svsInput(new PointCloud(cutWidth, cutHeight));
    for (int i = 0; i < cutWidth; ++i) {
        for (int j = 0; j < cutHeight; ++j) {
            svsInput->at(i, j) = cloud.at(cloud.width / 2 - cutWidth / 2 + i, cloud.height / 2 - cutHeight / 2 + j);
        }
    }
    ROS_INFO_STREAM("Cut " << svsInput->width << "x" << svsInput->height << " cloud");

    SVSParams params;
    SVSBuilder builder;
    builder.SetParams(params);
    builder.SetInputCloud(svsInput);
    builder.GenerateTrainingSet();
    builder.Learn();

    int const nTrainingObjects = builder.Objects->size();
    double const* alphas = builder.SVM().Alphas();

    svs::Alphas result;
    result.values.resize(nTrainingObjects);
    std::copy(alphas, alphas + nTrainingObjects, result.values.begin());

    alphaPublisher.publish(result);
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "svs_alpha_node");

    ros::NodeHandle n;
    n.getParam("/svs/cut/height", cutHeight);
    n.getParam("/svs/cut/width", cutWidth);
    ROS_INFO_STREAM("Cut height param is " << cutHeight);
    ROS_INFO_STREAM("Cut width param is " << cutWidth);

    alphaPublisher = n.advertise<svs::Alphas>("/svs/alphas", 1);
    cloudSubscriber = n.subscribe("/camera/rgb/points", 1, callback);

    ros::spin();
    return 0;
}
