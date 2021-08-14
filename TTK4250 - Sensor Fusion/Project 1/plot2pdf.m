path = './';
fig = gcf;
filename = 'track_lambda_1e-05';
fileformat = '-dpdf';

set(fig, 'Units', 'Inches');
pos1 = get(fig, 'Position');
    
fig_width = pos1(3);
fig_height = pos1(4);
    
set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches',...
        'PaperSize', [fig_width, fig_height]);    
   
if path(end) ~= '/'
    path(end + 1) = '/';
end

print(fig, strcat(path, filename), fileformat, '-r0');%, '-bestfit');