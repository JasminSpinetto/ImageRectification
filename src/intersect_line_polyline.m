function P = intersect_line_polyline(xL, yL, xx, yy)
% Intersect segment (xL,yL) with a polyline defined by (xx,yy).
% Returns Nx2 intersection points.

    x1 = xL(1); y1 = yL(1);
    x2 = xL(2); y2 = yL(2);

    P = zeros(0,2);

    for i = 1:(numel(xx)-1)
        x3 = xx(i);   y3 = yy(i);
        x4 = xx(i+1); y4 = yy(i+1);

        [hit, xi, yi] = segseg_intersect(x1,y1,x2,y2, x3,y3,x4,y4);
        if hit
            P(end+1,:) = [xi, yi]; %#ok<AGROW>
        end
    end

    % remove duplicates (line hits a vertex)
    if ~isempty(P)
        P = unique(round(P, 6), 'rows');
    end
end
