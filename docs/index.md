# LSGD Code Explanation
This page is mainly used for explaining the codes of distributed training library.

The general architecture is the following:
![alt text][dist_arch]

Each Node/Cluster/DevBox is usually equipped with 4 GPUs (workers). In the code we call all GPUs (workers) Node/Cluster/DevBox as a group.

## Batch Normaliztion (buffers in the codes)
One dilemma for distributed training is to deal with running mean and running variance of batch normalization (https://arxiv.org/abs/1502.03167). So far there is no way to perfectly synchronize running mean and running variance especially for asynchronous communication. Usually, when synchronizing we can ignore running mean and running variance so that each model (worker) keeps their own statistics. In our implementation, we average both running mean and running variance of all workers for every commumication period (As a side note, for PyTorch 1.0 version, synchronizing the empty tensors will cause an error).
