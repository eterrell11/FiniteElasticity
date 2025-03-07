figure("visible","off")
force_type = ["IV" "TR" "BF"];
nu = ["4" "49" "5"];
counter = 0;
for i = 1:3
	for j = 1:3
		error_table = readtable("error_table"+force_type(i)+nu(j)+".csv");
		error_array = table2array(error_table);
		if(size(error_array)~=0)
			subplot(3,3,counter+j)
			loglog(-(error_array(:,1)),error_array(:,2:end),"-*")
		end
	end
	counter = counter + 3;
end
%legend('Ev_{L2}','Ev_{L1}','Ev_{L\infty}','Ep_{L2}','Ep_{L1}','Ep_{L\infty}')
%xlabel("\Delta t")
%ylabel("Error magnitude")
%sgtitle("Error for velocity and pressure fields")
saveas(gcf,"error.fig")
quit
