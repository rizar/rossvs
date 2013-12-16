#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <svs/Alphas.h>
#include "components/svs.h"

ros::Publisher alphaPublisher;
ros::Publisher paramPublisher;
ros::Subscriber cloudSubscriber;

int cutWidth = 640;
int cutHeight = 480;
std::string paramPath;

void callback(sensor_msgs::PointCloud2ConstPtr msg) {
    ros::WallTime before = ros::WallTime::now();

    PointCloud cloud;
    pcl::fromROSMsg(*msg, cloud);
    ROS_INFO_STREAM("Received and converted " << cloud.width << "x" << cloud.height << " cloud");
    if (cloud.height == 1) {
        ROS_WARN_STREAM("Set dimensions explicitly");
        cloud.height = 480;
        cloud.width = 640;
    }

    PointCloud::Ptr svsInput(new PointCloud(cutWidth, cutHeight));
    for (int i = 0; i < cutWidth; ++i) {
        for (int j = 0; j < cutHeight; ++j) {
            svsInput->at(i, j) = cloud.at(cloud.width / 2 - cutWidth / 2 + i, cloud.height / 2 - cutHeight / 2 + j);
        }
    }
    ROS_INFO_STREAM("Cut " << svsInput->width << "x" << svsInput->height << " cloud");

    SVSParams params;
    if (paramPath.size()) {
        params.Load(paramPath.c_str());
    }
    // publish parameters
    {
        std_msgs::String paramStr;
        paramStr.data = params.ToString();
        paramPublisher.publish(paramStr);
    }


    SVSBuilder builder;
    builder.SetParams(params);
    builder.SetInputCloud(svsInput);
    builder.GenerateTrainingSet();
    builder.Learn();

    GridNeighbourModificationStrategy const& strat = *builder.Strategy;
    std::cout << "SVM converged in " << builder.SVM().Iteration << " iterations" << std::endl;
    std::cout << "SVM3D: " << builder.Objects->size() << " input vectors" << std::endl;
    std::cout << "SVM3D: " << builder.SVM().SVCount << " support vectors" << std::endl;
    std::cout << "SVM3D: " << builder.SVM().TargetFunction << " target function" << std::endl;
    std::cout << "SVM3D: " << builder.SVM().TouchedCount << " touched count" << std::endl;
    std::cout << "SVM3D: " << builder.SVM().MarginCrossCount << " margin cross count" << std::endl;
    std::cout << "GridStrategy: Number of cache misses: " << strat.NumCacheMisses << std::endl;
    std::cout << "GridStrategy: Number of optimization failures: " << strat.NumOptimizeFailures << std::endl;
    std::cout << "GridStrategy: Average number of neighbors: " <<
        static_cast<float>(strat.TotalNeighborsProcessed) / strat.NumNeighborsCalculations << std::endl;

    int const nTrainingObjects = builder.Objects->size();
    double const* alphas = builder.SVM().Alphas();

    svs::Alphas result;
    result.header = msg->header;
    result.values.resize(nTrainingObjects);
    std::copy(alphas, alphas + nTrainingObjects, result.values.begin());

    alphaPublisher.publish(result);
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "svs_alpha_node");

    ros::NodeHandle n;
    n.getParam("/svs/cut/height", cutHeight);
    n.getParam("/svs/cut/width", cutWidth);
    n.getParam("/svs/parampath", paramPath);
    ROS_INFO_STREAM("Cut height param is " << cutHeight);
    ROS_INFO_STREAM("Cut width param is " << cutWidth);
    ROS_INFO_STREAM("Parameters path is " << paramPath);

    alphaPublisher = n.advertise<svs::Alphas>("/svs/alphas", 1);
    paramPublisher = n.advertise<std_msgs::String>("/svs/alpha_node_params", 1);
    cloudSubscriber = n.subscribe("/camera/rgb/points", 10, callback);

    ros::spin();
    return 0;
}
