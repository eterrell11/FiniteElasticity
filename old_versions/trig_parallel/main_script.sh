#! /bin/sh

file=parameter_file.prm
mu_str=("4" "49" "5")
for m_str in ${mu_str[@]};
do
	sed -i 's/ratio=0.*/ratio=0.'$m_str'/' $file
	make run
done
