# NALU
Implementation of Neural Arithmetic Logic Units as discussed in https://arxiv.org/abs/1808.00508

The implementation here deviates from the paper when it comes to computing the gate variable **g**<br />
The paper enforces a dependence of **g** on the input **x** with the equation:
![equation](http://latex.codecogs.com/gif.latex?%5Cfn_jvn%20%24g%20%3D%20%5Csigma%20%5Cleft%20%28%20Gx%20%5Cright%20%29%24)<br />
However for most purposes the gating function is only dependant upon the task and not the input<br />
and can be learnt independantly of the input.<br />
This implementation uses ![equation](http://latex.codecogs.com/gif.latex?%5Cfn_jvn%20%24g%20%3D%20%5Csigma%28G%29%24) where **G** is a learnt scalar.
