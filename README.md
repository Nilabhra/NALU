# NALU
Implementation of **Neural Arithmetic Logic Units** as discussed in https://arxiv.org/abs/1808.00508

## This implementation
The implementation here deviates from the paper when it comes to computing the gate variable **g**<br />
The paper enforces a dependence of **g** on the input **x** with the equation:
![equation](http://latex.codecogs.com/gif.latex?%5Cfn_jvn%20%24g%20%3D%20%5Csigma%20%5Cleft%20%28%20Gx%20%5Cright%20%29%24)<br />
However for most purposes the gating function is only dependant upon the task and not the input<br />
and can be learnt independantly of the input.<br />
This implementation uses ![equation](http://latex.codecogs.com/gif.latex?%5Cfn_jvn%20%24g%20%3D%20%5Csigma%28G%29%24) where **G** is a learnt scalar.

For recurrent tasks, however, it does make sense to condition the gate value on the input.

## Limitations of a single NALU
- Can handle either add/subtract or mult/div operations but not a combination of both.
- For mult/div operations, it cannot handle negative targets as the mult/div gate output<br />
is the result of an exponentiation operation which always yeilds positive results.
- Power operations are only possible when the exponent is in the range of [0, 1].

## Advantages of using NALU
- The careful design of the mathematics ensure the learnt weights allow for both<br />
interpolation and extrapolation.

