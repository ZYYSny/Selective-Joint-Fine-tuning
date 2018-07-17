
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
5	0	96	4	2916	-2027.95	1789.98	
5	0	96	5	2916	-1384.63	1407.19	
5	0	96	6	2916	-1290.75	1034.77	
5	0	96	7	2916	-852.823	851.521	
5	0	96	8	2916	-870.249	770.438	
5	0	96	9	2916	-1537.28	1593.32	
5	0	96	10	2916	-407.13	270.886	
5	0	96	11	2916	-811.748	746.439	
5	0	96	12	2916	-1419.92	1370.32	
5	0	96	13	2916	-984.452	1135.34	
5	0	96	14	2916	-740.338	691.848	
5	0	96	15	2916	-1043.63	1131.85	
5	0	96	16	2916	-1191.07	1321.34	
5	0	96	17	2916	-1438.37	1405.2	
5	0	96	18	2916	-1429.19	1337.53	
5	0	96	19	2916	-1207.37	1203.34	

Format of  "caltech_alexnet_color_feature_stats_previous.txt"
   #The total input feature map groups | #The index of feature map groups | #Feature channels in current group | #Feature map index in current group | #Feature dim in every feature map | #Lower bound | #Upper bound
5	0	96	0	2916	-836.118	837.892	
5	0	96	1	2916	-3269.48	3340.07	
5	0	96	2	2916	-1012.25	1074.28	
5	0	96	3	2916	-1169.72	1107.96	
5	0	96	4	2916	-2027.95	1789.98	
5	0	96	5	2916	-1384.63	1407.19	
5	0	96	6	2916	-1290.75	1034.77	
5	0	96	7	2916	-852.823	851.521	
5	0	96	8	2916	-870.249	770.438	
5	0	96	9	2916	-1537.28	1593.32	
5	0	96	10	2916	-407.13	270.886	
5	0	96	11	2916	-811.748	746.439	
5	0	96	12	2916	-1419.92	1370.32	
5	0	96	13	2916	-984.452	1135.34	
5	0	96	14	2916	-740.338	691.848	
5	0	96	15	2916	-1043.63	1131.85	
5	0	96	16	2916	-1191.07	1321.34	
5	0	96	17	2916	-1438.37	1405.2	
5	0	96	18	2916	-1429.19	1337.53	
5	0	96	19	2916	-1207.37	1203.34	

