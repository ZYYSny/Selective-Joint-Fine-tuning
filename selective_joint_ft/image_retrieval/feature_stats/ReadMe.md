
# Feature Statistics Extraction

By [Weifeng Ge], [Yizhou Yu](http://i.cs.hku.hk/~yzyu/)

Department of Computer Science, The University of Hong Kong

The codes are not maintained for a long time. I will update the codes and make the nearest neighbor search more each using python.

1. Since we need to build histograms, so we need to know the value range of each channel in deep feature maps. Then you need to first run the ./get_the_stats_of_features.sh on ImageNet at first. You may edit the stats_deploy.prototxt like this.

       layer {
            name: "feature_stats"
            type: "FeatureStatistics"
            bottom: "conv1"
            bottom: "conv2"
            bottom: "conv3"
            bottom: "conv4"
            bottom: "conv5"
            top: "feat_stats"
            feature_stats_param{
              feature_statistics_path: "examples/cvpr2017/caltech-256/image_retrieval/feature_stats/caltech_alexnet_color_feature_stats.txt"
	      scales: 5
	      total_channels: 1376
	      saving_signal: 10
	      previous_statistics_path: "examples/cvpr2017/caltech-256/image_retrieval/feature_stats/caltech_alexnet_color_feature_stats_previous.txt"
	      new_feature_statistics_path: "examples/cvpr2017/caltech-256/image_retrieval/feature_stats/caltech_alexnet_color_feature_stats_domain.txt"
            }
            propagate_down: 0
            propagate_down: 0
            propagate_down: 0
            propagate_down: 0
            propagate_down: 0
      }
   
   Here caltech_alexnet_color_feature_stats.txt saves the upper and lower bounds of all channels in the five concated feature maps. 

2. Get the upper and lower bound of every feature channel, we run ./get_the_stats_of_features.sh again. Here caltech_alexnet_color_feature_stats_previous.txt stores the upper and lower bounds gotten in the previous round. caltech_alexnet_color_feature_stats_domain.txt stores the upper and lower bounds of every bin in the histogram. As desribed in the paper, the width of every bin is different from each other to make the histogram more discriminative.

3. [Some flaws in programming] Since these codes are not maintained, there are some flaws during usages. In the first step, we only want to get "caltech_alexnet_color_feature_stats.txt". We need to built some fake files like the format of "caltech_alexnet_color_feature_stats.txt". In the second step, we just rename "caltech_alexnet_color_feature_stats.txt" as "caltech_alexnet_color_feature_stats_previous.txt", then we finally get the bounds for every bin "caltech_alexnet_color_feature_stats_domain.txt".

 Format of  "caltech_alexnet_color_feature_stats.txt"  
 
   #The total input feature map groups | #The index of feature map groups | #Feature channels in current group | #Feature map index in current group | #Feature dim in every feature map | #Lower bound | #Upper bound  
   
5	0	96	0	2916	-836.118	837.892  

5	0	96	1	2916	-3269.48	3340.07  

5	0	96	2	2916	-1012.25	1074.28  

5	0	96	3	2916	-1169.72	1107.96   



Format of  "caltech_alexnet_color_feature_stats_previous.txt"  

   #The total input feature map groups | #The index of feature map groups | #Feature channels in current group | #Feature map index in current group | #Feature dim in every feature map | #Lower bound | #Upper bound  
   
5	0	96	0	2916	-836.118	837.892	  

5	0	96	1	2916	-3269.48	3340.07  

5	0	96	2	2916	-1012.25	1074.28  

5	0	96	3	2916	-1169.72	1107.96	  


