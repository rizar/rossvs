#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <svs/Alphas.h>
#include "components/svs.h"
#include "components/visualization.h"

std::string paramPath;

ros::Publisher adjGradPub;

PointCloud::Ptr repainted(PointCloud const& cloud, std::function<Color (int, int)> colorMap) {
    PointCloud::Ptr result(new PointCloud(cloud));
    for (int i = 0; i < cloud.width; ++i) {
        for (int j = 0; j < cloud.height; ++j) {
            PointType & p = result->at(i, j);
            Color c = colorMap(i, j);
            p.r = c.R;
            p.g = c.G;
            p.b = c.B;
        }
    }
    return result;
}

void callback(
        svs::AlphasConstPtr alphaMsg,
        sensor_msgs::PointCloud2ConstPtr cloudMsg)
{
    // no cutting in this node yet
    ros::WallTime before = ros::WallTime::now();

    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloudMsg, *cloud);
    ROS_INFO_STREAM("Received and converted " << cloud->width << "x" << cloud->height << " cloud ");
    if (cloud->height == 1) {
        ROS_WARN_STREAM("Set dimensions explicitly");
        cloud->height = 480;
        cloud->width = 640;
    }

    SVSParams params;
    if (paramPath.size()) {
        params.Load(paramPath.c_str());
    }

    SVSBuilder builder;
    builder.SetParams(params);
    builder.SetInputCloud(cloud);
    builder.GenerateTrainingSet();
    builder.InitSVM(alphaMsg->values);
    builder.CalcGradients();

    {
        std::vector<float> agn;
        builder.ToImageLayout(builder.AdjustedGradientNorms, &agn);
        float const maxAgn = *std::max_element(agn.begin(), agn.end());
        PointCloud::Ptr agnCloud = repainted(*cloud, [&cloud, &agn, maxAgn] (int i, int j) {
                    int const idx = j * cloud->width + i;
                    float const relAgn = agn[idx] / maxAgn;
                    return Color::fromRelative(relAgn, 0, 1 - relAgn);
                });

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*agnCloud, msg);
        msg.header = cloudMsg->header;
        adjGradPub.publish(msg);
    }

    ROS_INFO_STREAM("Job done in " << ros::WallTime::now() - before);
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "svs_vis_node");

    ros::NodeHandle n;
    n.getParam("/svs/parampath", paramPath);
    ROS_INFO_STREAM("Parameters path is " << paramPath);

    adjGradPub = n.advertise<sensor_msgs::PointCloud2>("svs/agn_point_cloud", 10);

    message_filters::Subscriber<svs::Alphas> alphaSubscriber(n,"/svs/alphas", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloudSubscriber(n, "/camera/rgb/points", 10);
    message_filters::TimeSynchronizer<svs::Alphas, sensor_msgs::PointCloud2> synchronizer
        (alphaSubscriber, cloudSubscriber, 10);
    synchronizer.registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();
    return 0;
}

