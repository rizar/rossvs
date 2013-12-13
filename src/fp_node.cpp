#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <pcl/range_image/range_image_planar.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/range_image_border_extractor.h>

#include <svs/Alphas.h>
#include "components/svs.h"
#include "components/searcher.h"

int numFeaturePoints;
ros::Publisher svsPub;
ros::Publisher narfPub;
std::string paramPath;

void extractAndPublishSVS(svs::AlphasConstPtr alphaMsg,
        sensor_msgs::PointCloud2ConstPtr cloudMsg,
        PointCloud::Ptr cloud)
{
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
    builder.FeaturePointSearch();

    {
        // make it white
        PointCloud::Ptr fp = builder.FeaturePoints;
        for (int i = 0; i < fp->size(); ++i) {
            fp->at(i).r = 255;
            fp->at(i).g = 255;
            fp->at(i).b = 255;
        }

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*fp, msg);
        msg.header = cloudMsg->header;
        svsPub.publish(msg);
    }
}

void extractAndPublishNARF(
        sensor_msgs::PointCloud2ConstPtr cloudMsg,
        PointCloud::Ptr const& cloud) {
    pcl::RangeImagePlanar rip;
    pcl::RangeImageBorderExtractor ribe;
    rip.createFromPointCloudWithFixedSize(*cloud, cloud->width, cloud->height,
            319.5, 239.5, 525.0, 525.0, static_cast<Eigen::Affine3f>(Eigen::Translation3f(0.0, 0.0, 0.0)));
    rip.setUnseenToMaxRange();
    ROS_INFO_STREAM("Built range image " << rip.width << "x" << rip.height);

    pcl::NarfKeypoint narf;
    narf.setRangeImageBorderExtractor(&ribe);
    narf.setRangeImage(&rip);
    narf.getParameters().support_size = 0.1;
    narf.getParameters().use_recursive_scale_reduction = true;
    narf.getParameters().calculate_sparse_interest_image = true;

    pcl::PointCloud<int> indices;
    narf.compute(indices);

    PointCloud result;
    for (int i = 0; i < indices.size(); ++i) {
        result.push_back(cloud->at(indices[i]));
    }

    {
        sensor_msgs::PointCloud2 resMsg;
        pcl::toROSMsg(result, resMsg);
        resMsg.header = cloudMsg->header;
        narfPub.publish(resMsg);
    }

    ROS_INFO_STREAM("Published " << indices.size() << " narf feature points");
}

void callback(
        svs::AlphasConstPtr alphaMsg,
        sensor_msgs::PointCloud2ConstPtr cloudMsg)
{
// no cutting in this node yet
    ros::WallTime before = ros::WallTime::now();

    PointCloud::Ptr cloud(new PointCloud);
    pcl::fromROSMsg(*cloudMsg, *cloud);
    ROS_INFO_STREAM("Received and converted " << cloud->width << "x" << cloud->height
                    << " cloud in frame" << cloudMsg->header.frame_id);

    extractAndPublishSVS(alphaMsg, cloudMsg, cloud);
    extractAndPublishNARF(cloudMsg, cloud);

    ROS_INFO_STREAM("Job done in " << ros::WallTime::now() - before);
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "svs_alpha_node");

    ros::NodeHandle n;
    n.getParam("/svs/parampath", paramPath);
    ROS_INFO_STREAM("Parameters path is " << paramPath);

    svsPub = n.advertise<sensor_msgs::PointCloud2>("/svs/fp", 1);
    narfPub = n.advertise<sensor_msgs::PointCloud2>("/svs/narf", 1);

    message_filters::Subscriber<svs::Alphas> alphaSubscriber(n, "/svs/alphas", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloudSubscriber(n, "/camera/rgb/points", 10);
    message_filters::TimeSynchronizer<svs::Alphas, sensor_msgs::PointCloud2> synchronizer
        (alphaSubscriber, cloudSubscriber, 10);
    synchronizer.registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();
    return 0;
}
