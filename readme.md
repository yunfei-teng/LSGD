# LSGD
This is the repository for paper LSGD (https://arxiv.org/abs/1905.10395)

## Requirements
1. Install PyTorch from PyTorch official website (https://pytorch.org/)

2. Type ```nvidia-smi``` in terminal to check nvidia driver and cuda compatibility (https://docs.nvidia.com/deploy/cuda-compatibility/)

3. install termcolor by ```pip install termcolor```

## Instruction
1. Assume you have **n** GPU nodes for distributed training. On the first GPU node, type command ```ifconfig``` in terminal to check its IP address;

2. Open the bash file ```lsgd.sh```: fill in *ip_addr* with the IP address you obtained from step 1 and *num_swarms* with the total number of GPU nodes you have (i.e. *num_swarms=n*);

3. On *k*-th GPU note, in terminal type *bash lsgd.sh k-1* (i.e. index starts from 0) to run the codes.
