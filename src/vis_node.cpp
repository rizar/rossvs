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
ros::Publisher svPub;
ros::Publisher posPub;
ros::Publisher negPub;

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

void publishPointCloud(ros::Publisher * pub, PointCloud const& pc, std_msgs::Header const& header) {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(pc, msg);
    msg.header = header;
    pub->publish(msg);
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

    // publish positive and negative examples in separate clouds
    {
        PointCloud pos;
        PointCloud neg;
        for (int i = 0; i < builder.Objects->size(); ++i) {
            if (builder.Labels[i] == +1) {
                pos.push_back(builder.Objects->at(i));
            } else {
                neg.push_back(builder.Objects->at(i));
            }
        }

        {
            publishPointCloud(&posPub, pos, cloudMsg->header);
            publishPointCloud(&negPub, neg, cloudMsg->header);
        }
    }

    // publish support vector cloud
    {
        PointCloud sv;
        for (int i = 0; i < builder.Objects->size(); ++i) {
            PointType p = builder.Objects->at(i);
            if (alphaMsg->values[i] > 0.0) {
                p.r = p.g = p.b = 0;
                if (builder.Labels[i] == +1) {
                    p.r = 255 * (alphaMsg->values[i] / params.MaxAlpha);
                } else {
                    p.b = 255 * (alphaMsg->values[i] / params.MaxAlpha);
                }
                sv.push_back(p);
            }
        }

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(sv, msg);
        msg.header = cloudMsg->header;
        svPub.publish(msg);

        ROS_INFO_STREAM("Published " << sv.size() << " support vectors");
    }

    // publish adjusted gradient norm
    {
        std::vector<float> agn;
        builder.ToImageLayout(builder.AdjustedGradientNorms, &agn);
        float const maxAgn = *std::max_element(agn.begin(), agn.end());
        PointCloud::Ptr agnCloud = repainted(*cloud, [&cloud, &agn, maxAgn] (int i, int j) {
                    int const idx = j * cloud->width + i;
                    float const relAgn = agn[idx] / maxAgn;
                    float const remap = std::max(0.0, 1 + 0.2 * log(relAgn));
                    return Color::fromRelative(remap, 0, 1 - remap);
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
    svPub = n.advertise<sensor_msgs::PointCloud2>("svs/sv", 10);
    posPub = n.advertise<sensor_msgs::PointCloud2>("svs/positive", 10);
    negPub = n.advertise<sensor_msgs::PointCloud2>("svs/negative", 10);

    message_filters::Subscriber<svs::Alphas> alphaSubscriber(n,"/svs/alphas", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloudSubscriber(n, "/camera/rgb/points", 10);
    message_filters::TimeSynchronizer<svs::Alphas, sensor_msgs::PointCloud2> synchronizer
        (alphaSubscriber, cloudSubscriber, 10);
    synchronizer.registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();
    return 0;
}

