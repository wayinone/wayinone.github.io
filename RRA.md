---
layout: default2
title: RRA
permalink: /RRA
---
# Prototype Learning: Generalized K-means
---
Wei-Ying Wang and Stuart Geman, 2017

How to learn the inherent structures of data? Especially, they are non-labeled (this makes it unsupervised learning problem). Our algorithm can do it pretty well! In the right plot, after we found 3 structures (AKA prototypes) we summarized the data point according to its nearest structure, and colored it by different colors.

| <center>Original data</center> | <center>Proposed Method: Structures learned</center> | 
| ---------------------------- |:----------------------------:| 
| <center><img src="ContiReading/RRA/Triangle_Cor.png" style="width: 300px;" /></center> | <center><img src="ContiReading/RRA/Triangle_Cor_RRA_Random.png" style="width: 300px;" /></center>  | 

Another example in 3D:

| <center>Original data</center> | <center>Proposed Method: Structures learned</center> | 
| ---------------------------- |:----------------------------:| 
| <center><img src="ContiReading/RRA/plane 3D_0.png" style="width: 300px;" /></center> | <center><img src="ContiReading/RRA/plane 3D_exact_m2_0.png" style="width: 300px;" /></center>  | 

Our idea is to generalize the loss function of K-means (minimizing it is an NP-hard problem) and minimize it. However, direct generalizing the idea of K-means algorithm is NOT working. Instead, we characterize a property of the minimizer of the loss function and built an algorithm that approximate it. In nearly all our experiments, the algorithm find the exact minimizer of our loss function.

The traditional way of finding these structures is using EM algorithm with Gaussian model, which is very similar to that of 
K-means, however, it fails horribly once we add some outliers to the data. 

| <center>Original data</center> | <center>EM algorithm</center> | <center>Proposed Method</center> | 
---|---|---
| <center><img src="ContiReading/RRA/EM_compare_original_points.png" style="width: 250px;" /></center> | <center><img src="ContiReading/RRA/EM_2lines_40outlier_n100.png" style="width: 250px;" /></center>  | <center><img src="ContiReading/RRA/RRA_b08_l30_2lines_40outlier_n100.png" style="width: 250px;" /></center> |

Let's look another example, which has more outliers, to demonstrate the robust nature of our method.

| <center>Original data</center> | <center>Proposed Method</center> | 
| ---------------------------- |:----------------------------:| 
| <center><img src="ContiReading/RRA/XshapeCor.png" style="width: 300px;" /></center> |<center><img src="ContiReading/RRA/XshapeCor_RRA_Local_Best.png" style="width: 300px;" /></center>  | 

How about finding multiple *different*  structures? Our algotihm can do it with some little "twist." The "twist" uses the idea of minimal volume of a cluster. In the following plots, the right one is the structure our algorithm found, we put two circle and a rectangle to give reader the idea of minimum volumes. For example: the volume of a point cluster (cluster that closes to a point structure) is defined as an area of a circle. 

| <center>Original data</center> | <center>Proposed Method</center> | 
---|---
| <center><img src="ContiReading/RRA/MV_pct_n130.png" style="width: 300px;" /></center> |<center><img src="ContiReading/RRA/MV_pct_n130_70th_ptile_m3.png" style="width: 300px;" /></center>  | 

An example in 3D:

| <center>Original data</center> | <center>Proposed Method</center> | 
---|---
| <center><img src="ContiReading/RRA/MV_3Dmix_n65.png" style="width: 300px;" /></center> |<center><img src="ContiReading/RRA/MV_3Dmix_n65_70th_ptile_m3.png" style="width: 300px;" /></center>  | 

By the way, we have a rule of thumb to determine the number of structures.