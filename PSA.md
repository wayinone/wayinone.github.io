---
layout: default2
title: Prediction Shift Algorithm (PSA)
permalink: /PSA
---
# New Tool for Data Visualization/Dimension reduction
---
Wei-Ying Wang and Stuart Geman, 2017

PCA is probably the oldest tool to achieve dimension reduction. If you run PCA on data (or say, do spectrum decomposition on the empirical covariance matrix), the direction of the largest eigenvector (which corresponds to the largest eigenvalue) will be the vector that achieves the widest spread of the data. That is, if you project your data to the vector, you have the largest variance among all other vectors of the same dimension.

Stu and I developed a very simple and effective algorithm, called PSA. The algorithm combines mean-shift and PCA idea. In particular, we can adjust a parameter to achieve data visualization and dimension reduction.

Data Visaulization with PSA: The operation took less than 1 second in my python script for 400 3D data points. In the left is the original points, and in the right is the result of PSA after 6 iterations. It captures the "spiral" shape of the data.

<center>
	<img src="ContiReading/PSA/line_model 20 NN 1.5 shrinkage.png" style="width: 600px;" />
</center>

Dimension Reduction with PSA: Let test our reduction algorithm with the famous S-shped data used in LLE (Local Linear Embedding). (See [here](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html).) In the left image is the original data, and in the right is the result of PSA after 100 iteration. Note that the color of each points are corresponding to each other. This took about 5 second for this 1000 data. A great way to understand how it work is through the video, as we can see it unfolds the complicated structure of data.

<center>
	<img src="ContiReading/PSA/PSA on 3D S data.png" style="width: 600px;" />
</center>

<center>
<video width="320" height="320" controls>
  <source src="\ContiReading/PSA/movie PSA on 3D S data.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
</center>

We also ran the algorithm for our previous data, and reduced it to one dimension.

<center>
<video width="320" height="320" controls>
  <source src="\ContiReading/PSA/helix0.95shrikage.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
</center>