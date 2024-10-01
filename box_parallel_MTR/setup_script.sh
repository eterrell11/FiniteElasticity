#module load gcc/11.2.0
#module load matlab
#module load openmpi/4.1.5-gcc_11.2.0

sh puller.sh
cp ~/FiniteElasticityGit/src/FiniteElasticity/box_parallel_mtr/box_parallel_mtr.cc .
#source /users/t/e/terrell1/longleaf/sfw/autoibamr/install-release
rm compile_commands.json cmake_install.cmake Makefile
rm -r CMakeFiles
cmake -DDEAL_II_DIR=~/longleaf/sfw/dealii/build .


make release
make