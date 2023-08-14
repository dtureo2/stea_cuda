# stea_cuda
A very old project, cryptoanalysis into stea (Simple Tiny Encryption Algorithm) using CUDA capabilities. All the instructions are in spanish, for now. I created in 2009 for my final bachelor project.

To compile nvcc -o cuda stea.cu' or nvcc -o gpu_ataque -Xptxas "-v" -maxrregcount=10 stea.cu

(in the future there's going to be a english version and also a non brute force attack code)
