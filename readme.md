# LSGD
This is the code repository for paper LSGD along with explanation (https://yunfei-teng.github.io/LSGD/)

## Requirements
1. Install PyTorch from official website (https://pytorch.org/). This is one of the most popular deep learning tools.

2. install termcolor by ```pip install termcolor```. This colorizes the output from terminal.

## Instruction
1. Assume you have **n** GPU nodes for distributed training. On the first GPU node, type command ```ifconfig``` in terminal to check its IP address;

2. Open the bash file ```lsgd.sh```: fill in *ip_addr* with the IP address you obtained from step 1 and *num_swarms* with the total number of GPU nodes you have (i.e. *num_swarms=n*);

3. On *k*-th GPU note, in terminal type *bash lsgd.sh k-1* (i.e. index starts from 0) to run the codes.

## Possible Issues
1. Type ```nvidia-smi``` in terminal to check nvidia driver and cuda compatibility (https://docs.nvidia.com/deploy/cuda-compatibility/)

2. Check consistency of PyTorch versions across the machines
