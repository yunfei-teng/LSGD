# LSGD
This page is mainly used for explaining the codes of distributed training library of LSGD.

The general architecture is the following:
![](./dist_arch)

At the beginning of each iteration, the local worker sends out a request to its local server and
then the local server passes on the worker’s request to the global server. The global server checks the
current status and replies to the local server. The local server passes on the global server’s message
to the worker. Finally, depending on the message from the global server, the worker will choose to
follow the local training or distributed training scheme.

Each Node/Cluster/DevBox is usually equipped with 4 GPUs (workers). In the code we call all GPUs (workers) Node/Cluster/DevBox as a group.

## Batch Normaliztion
One dilemma for distributed training is to deal with running mean and running variance of [batch normalization](https://arxiv.org/abs/1502.03167). Based on our knowledge, there is no perfect solution to synchronize running mean and running variance especially for asynchronous communication so far. Usually, when synchronizing we can ignore running mean and running variance so that each model (worker) keeps their own statistics. In our implementation, we average both running mean and running variance of all workers for every commumication period (As a side note, for PyTorch 1.0 version, synchronizing the empty tensors will cause an error, so make sure to comment out the parts of synchronizing the buffers).
