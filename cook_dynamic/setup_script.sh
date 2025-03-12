
rm CMakeCache.txt
sh puller.sh
cp ~/FiniteElasticityGit/src/FiniteElasticity/cook_parallel_refine/cook_dynamic.cc .
rm compile_commands.json cmake_install.cmake Makefile
rm -r CMakeFiles
cmake -DDEAL_II_DIR=~/longleaf/sfw/dealii/build .


make release
make