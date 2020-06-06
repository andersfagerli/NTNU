function draw_frame(K, T, len)
    uv0 = project(K, T*[0 0 0 1]');
    uvx = project(K, T*[len 0 0 1]');
    uvy = project(K, T*[0 len 0 1]');
    uvz = project(K, T*[0 0 len 1]');
    plot([uv0(1) uvx(1)], [uv0(2) uvx(2)], 'color', '#cc4422', 'linewidth', 2);
    plot([uv0(1) uvy(1)], [uv0(2) uvy(2)], 'color', '#11ff33', 'linewidth', 2);
    plot([uv0(1) uvz(1)], [uv0(2) uvz(2)], 'color', '#3366ff', 'linewidth', 2);
    text(uvx(1), uvx(2), 'x');
    text(uvy(1), uvy(2), 'y');
    text(uvz(1), uvz(2), 'z');
end