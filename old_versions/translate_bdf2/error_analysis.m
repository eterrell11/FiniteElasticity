clear all
close all
error_table = readtable('error_table.csv');
error_array = table2array(error_table);

if(size(error_array)~=0)
    efig = figure;
    loglog(-(error_array(:,1)),error_array(:,2:end),"-*")
    legend('Ev_{L2}','Ev_{L1}','Ev_{L\infty}','Ep_{L2}','Ep_{L1}','Ep_{L\infty}')
    xlabel("\Delta t")
    ylabel("Error magnitude")
    title("Error for velocity and pressure fields")
    saveas(gcf,"error.png")
end
close all
