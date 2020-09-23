# KCC
Parallel implementation of k-Closest Circles (KCC) algorithm [1] using Nvidia CUDA and C.

## Description
Given a set of points, this algorithm groups them around circles such that the sum of distances from the data points and appropriate closest circles is minimized.
The problem is solved by application of the center-based clustering method.

<img src="https://github.com/michael2393/KCC/blob/master/images/event_example.png" width="400" height="400">

## Input data
The algorithm takes as input multiple sets of data points in the 2-dimensional euclidean space (called 'events'), and for each event it runs the k-closest circles algorithm.
The input data file can be given as argument from the command line. Otherwise, the default location of input data is <code>Input/batch00.dat</code>.


## Efficiency
The algorithm splits the input into multiple blocks and threads and achieves 32 to 55 times speedup compared to a sequential implementation of KCC (in the following system).

<table>
<thead>
</thead>
<tbody>
<tr>
<td align="left"><strong>CPU</strong></td>
<td align="left">i7 4710MQ (3.4 GHz)</td>
</tr>
<tr>
<td align="left"><strong>GPU</strong></td>
<td align="left">Nvidia 840M</td>
</tr>
 <tr>
<td align="left"><strong>RAM</strong></td>
<td align="left">16 GB</td>
</tr>
</tbody>
</table>


## Results
The results are stored in a "Results.txt" file as follows.

```
Event: i -> numCircles = C
Circle 1 => (x_1, y_1) - radius_1
Circle 2 => (x_2, y_2) - radius_2
. . .
```

They are also saved in raw format.
Initially there is the number of the event and the number of circles, and it is followed by the cycles (x, y, and its radius).

```
1, C_1
X_1, Y_1, Radius_1
X_2, Y_2, Radius_2
. . .
2, C_2
X_1, Y_1, Radius_1
X_2, Y_2, Radius_2
. . .
```

## References
[1] <a href="https://hrcak.srce.hr/ojs/index.php/crorr/article/view/2216">Data clustering for circle detection, Tomislav Marošević, Croatian Operational Research Review, 2014</a>
