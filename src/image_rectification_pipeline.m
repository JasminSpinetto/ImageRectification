%% Image Analysis and Computer Vision Homework by Jasmin Spinetto
% Year 2025/26, Prof. Vincenzo Caglioti

% This code implements the solution to theoretical questions 1 to 6 and the
% 3D rendering of one of the ogival ribs.

clear all 
close all
clc

%% Load image and extracted features
I = imread(fullfile('data', 'raw', 'san_maurizio.jpg'));

FNT_SZ = 20;

load(fullfile('data', 'processed', 'san_maurizio_features.mat'));

%==========================================================================
% Convert each selected segment (two endpoints in image coordinates) into a
% homogeneous 2D line representation l = [l1 l2 l3]^T such that:
%     l1*x + l2*y + l3 = 0
%
% Each segment is provided as a row: [x1 y1 x2 y2].
% We lift endpoints to homogeneous points a=[x1 y1 1]^T, b=[x2 y2 1]^T and
% compute the line through them via the cross product:
%     l = a x b
%
% We then normalize the line by ||[l1 l2]|| so that its direction/scale is
% consistent across lines (useful for averaging / least-squares vanishing
% point / robust estimation). The sign remains arbitrary (l and -l are same).
%==========================================================================

% vertical lines
numSegments_v = size(endPointsV,1);
verticalLines = zeros(numSegments_v,3);

for i = 1:numSegments_v
    a = [endPointsV(i,1:2) 1]';
    b = [endPointsV(i,3:4) 1]';
    l = cross(a,b);
    verticalLines(i,:) = (l ./ norm(l(1:2)))';
end

%horizontal lines
numSegments_h = size(endPointsH,1);
horizontalLines = zeros(numSegments_h,3);

for i = 1:numSegments_h
    a = [endPointsH(i,1:2) 1]';
    b = [endPointsH(i,3:4) 1]';
    l = cross(a,b);
    horizontalLines(i,:) = (l ./ norm(l(1:2)))';
end

% cylinder axis parallel lines
numSegments_ax = size(endPointsAX,1);
AxisLines = zeros(numSegments_ax,3);

for i = 1:numSegments_ax
    a = [endPointsAX(i,1:2) 1]';
    b = [endPointsAX(i,3:4) 1]';
    l = cross(a,b);
    AxisLines(i,:) = (l ./ norm(l(1:2)))';
end

%% 1: FIND VANISHING LINE OF ANY PLANE PERPENDICULAR TO CYLINDER AXIS
% ==========================================================================
% The rectification plane contains two dominant, orthogonal direction families:
%     (i) "vertical" direction lines in the scene
%     (ii) "horizontal" direction lines in the scene
% Each family converges to a vanishing point in the image. We find it
% through SVD since the line selection has been performed manually and is
% consequently imprecise, possibly leading to multiple intersection points
% between lines of the same parallel set.
% The vanishing line of the plane is the line through the two VPs:
% l = VP_vertical × VP_horizontal   (cross product in homogeneous coordinates)
%==========================================================================
[u,s,v]=svd(verticalLines); %solving the homogeneus system L*VP=0 through SVD
V_vert = v(:,end); % Last column minimizes algebraic error
V_vert = V_vert/V_vert(3); %normalizing

[u,s,v]=svd(horizontalLines);
V_hor = v(:,end);
V_hor = V_hor/V_hor(3);

% Calculate the vanishing line by taking the cross product of the vanishing points
vanishingLine = cross(V_vert, V_hor);
vanishingLine = vanishingLine / norm(vanishingLine(1:2)); % Normalize the vanishing line

figure(1);
imshow(I); hold on;

plot(V_vert(1), V_vert(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(V_hor(1),  V_hor(2),  'bo', 'MarkerSize', 10, 'LineWidth', 2);

% since VPs stand outside the image, "imwarp" doesn't show them
% automatically, we must force it:
W = size(I,2); H = size(I,1);

% points to include
pts = [1 W  V_vert(1) V_hor(1);
       1 H  V_vert(2) V_hor(2)];

% robust bounds
xmin = min(pts(1,:)); xmax = max(pts(1,:));
ymin = min(pts(2,:)); ymax = max(pts(2,:));

xlim([xmin xmax]); ylim([ymin ymax]);

% Plot vanishing line l = [a b c]^T using the line equation:
%    a x + b y + c = 0  ->  y = -(a x + c)/b
lineX = [1, size(I,2)];
lineY = -(vanishingLine(1)*lineX + vanishingLine(3)) / vanishingLine(2);
plot(lineX, lineY, 'y-', 'LineWidth', 2);

legend({'VP vertical','VP horizontal','vanishing line'});

%% 2: VANISHING POINT OF DIRECTION OF CYLINDER AXIS
% ==========================================================================
% All image lines parallel to the cylinder axis intersect at one vanishing
% point (actually, here also due to human selection error they will
% intersect in multiple points, we handle this through SVD).
% As before, each image line l_i satisfies:  l_i^T * VP_axis = 0.
% ==========================================================================
[~,~,v] = svd(AxisLines);
V_axis = v(:,end);
V_axis = V_axis / V_axis(3);

% update axis limits to contain also V_axis
pts = [1 W  V_vert(1) V_hor(1) V_axis(1);
       1 H  V_vert(2) V_hor(2) V_axis(2)];

xmin = min(pts(1,:)); xmax = max(pts(1,:));
ymin = min(pts(2,:)); ymax = max(pts(2,:));

xlim([xmin xmax]); ylim([ymin ymax]);

% Plot the vanishing point of the cylinder axis
plot(V_axis(1), V_axis(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
legend({'VP vertical','VP horizontal','vanishing line', 'VP cylinder axis'});

hold off;

%% 3 - IMAGE RECTIFICATION
% Performed through the stratified method.

%% 3A: Affine Rectification
% ==========================================================================
% Remove projective distortion of the plane by sending its vanishing line
% to the line at infinity. After this transformation, parallel lines in
% the plane become parallel again.
% The vanishing line l = [a b c]^T represents the image of the line at infinity.
% We build a homography H_A such that:
% H_A^{-T} * l  ∝  [0 0 1]^T
% ==========================================================================
l = vanishingLine(:); %already normalized

% build affine matrix
HA = [1 0 0;
      0 1 0;
      l(1)/l(3)  l(2)/l(3)  1];

% sanity check: should output [0 0 1]^T
fprintf('The vanishing line is mapped to:\n');
disp(inv(HA)'*vanishingLine);

% Apply affine rectification
tform = projective2d(HA');
I_Affine= imwarp(I,tform);

figure(2);
imshow(I_Affine);

% Limit view to remove useless black regions
xlim([0 2500]);
ylim([0 2500]);

%% 3B: Metric Rectification
% Exploit the image of a conic as metric contraint. 

%% Select relevant points manually
% ==========================================================================
% Extract two symmetric ogival ribs.
% ==========================================================================
% Select a reference point N_{ij} on the vault which defines the
% orthogonal plane we rectify, then select two symmetric ogival ribs (of 
% two families a_i and b_j) crossing N_{ij}
% by selecting points on them to fit to curves

figure(3); imshow(I); axis on; hold on;

title('Select N_{ij}');
[x, y] = getpts();

% Save point in homogeneous coordinates
N_ij = [x(1); y(1); 1];

% plot point
plot(N_ij(1), N_ij(2), 'yo', 'MarkerSize',8,'LineWidth',2);
text(N_ij(1)+50, N_ij(2)+30, '$N_{ij}$', ...
     'Color','y','FontSize',12,'FontWeight','bold', ...
     'Interpreter','latex');


% select points on symmetric ogival ribs
figure(3); hold on;

% fix number of points to select
K = 8; % by trying different values, I found 8 points were enough to obtain
% a sufficiently well-fitted curve

% Selection of ogival rib of family a_i: manually sample points along the rib crossing N_{ij}.
% These points are later used to reconstruct the full curve.
title('Select 8 points on the ogival rib crossing N_{ij} of family a_i');
arc1 = zeros(3, K);
for k = 1:K
    [x1,y1] = ginput(1);
    arc1(:,k) = [x1; y1; 1]; %homogeneus coordinates
    plot(x1,y1,'ro','MarkerSize',8,'LineWidth',2);
    text(x1+5,y1, sprintf('a_{%d}',k), ...
         'Color','r','FontSize',12,'FontWeight','bold');
end
disp('Press enter to continue'); pause;

% Selection of ogival rib of family b_i: manually sample points along the rib crossing N_{ij}.
% These points are later used to reconstruct the full curve.
title('Select 8 points on the ogival rib crossing N_{ij} of family b_i');
arc2 = zeros(3, K);
for k = 1:K
    [x1,y1] = ginput(1);
    arc2(:,k) = [x1; y1; 1]; %homogeneus coordinates
    plot(x1,y1,'bo','MarkerSize',8,'LineWidth',2);
    text(x1+5,y1, sprintf('b_{%d}',k), ...
         'Color','b','FontSize',12,'FontWeight','bold');
end

hold off;

%PCHIP fit (shape-preserving) for both arcs

% Parametric PCHIP fit for both arcs (robust for non-function curves)
% The ogival ribs are not functions y = f(x), so we use a parametric form:
% x = x(t),  y = y(t)
% Parameter t is chosen as cumulative chord length to ensure robustness
% to uneven or unordered user clicks. 

% Extract points
x1 = arc1(1,:)';  y1 = arc1(2,:)';
x2 = arc2(1,:)';  y2 = arc2(2,:)';

% Order points along the curve, needed because the b_j rib curves above itself:
% use cumulative chord-length as parameter
t1 = [0; cumsum(hypot(diff(x1), diff(y1)))];
t2 = [0; cumsum(hypot(diff(x2), diff(y2)))];

% Remove duplicate parameter values (can happen if two identical clicks)
[t1u, idx1] = unique(t1, 'stable');  x1u = x1(idx1);  y1u = y1(idx1);
[t2u, idx2] = unique(t2, 'stable');  x2u = x2(idx2);  y2u = y2(idx2);

% Dense parameter grids for smooth visualization
tt1 = linspace(t1u(1), t1u(end), 400);
tt2 = linspace(t2u(1), t2u(end), 400);

% PCHIP in parametric form
xx1 = interp1(t1u, x1u, tt1, 'pchip');
yy1 = interp1(t1u, y1u, tt1, 'pchip');

xx2 = interp1(t2u, x2u, tt2, 'pchip');
yy2 = interp1(t2u, y2u, tt2, 'pchip');

% Plot
figure(3); hold on;
plot(xx1, yy1, 'r-', 'LineWidth', 2);
plot(xx2, yy2, 'b-', 'LineWidth', 2);
hold off;

%% Find intersections
% ==========================================================================
% We sample a family of image lines passing through the vanishing point of the
% cylinder axis V_axis. For each line direction θ, we compute its intersections
% with the two fitted rib polylines (arc1 and arc2). The resulting pairs (P1,P2)
% define corresponding (symmetric w.r.t. rectifying plane) points on the two symmetric ribs.
% ==========================================================================
% Extract inhomogeneus coordinates
Vx = V_axis(1);
Vy = V_axis(2);

% Ensure curve samples are row vectors (polyline format expected by
% intersection routine)
xx1 = xx1(:)'; yy1 = yy1(:)';
xx2 = xx2(:)'; yy2 = yy2(:)';

% Generate Ntheta directions and create a long segment through V_axis for each θ.
Ntheta = 200;
thetas = linspace(-pi/4, pi/2, Ntheta); %angular range sweeped
T = 1e4;  % long segment length

% Containers for intersection pairs (inhomogeneous coordinates)
P1_list = [];
P2_list = [];
theta_list = [];

figure(4); imshow(I); hold on;

for it = 1:numel(thetas)
    th = thetas(it);
    % Unit direction vector for the current ray/line
    dx = cos(th); dy = sin(th);

    % long segment through V with direction dx
    xL = [Vx - T*dx, Vx + T*dx];
    yL = [Vy - T*dy, Vy + T*dy];

    % intersections with each arc polyline (using self-defined function)
    P1_all = intersect_line_polyline(xL, yL, xx1, yy1);
    P2_all = intersect_line_polyline(xL, yL, xx2, yy2);

    % only save points if the line intersects both arcs
    if isempty(P1_all) || isempty(P2_all)
        continue;
    end

    % pick intersection closest to V
    d1 = hypot(P1_all(:,1) - Vx, P1_all(:,2) - Vy);
    [~, j1] = min(d1);
    P1 = P1_all(j1,:);

    d2 = hypot(P2_all(:,1) - Vx, P2_all(:,2) - Vy);
    [~, j2] = min(d2);
    P2 = P2_all(j2,:);

    % draw the segment between intersections
    plot([P1(1) P2(1)], [P1(2) P2(2)], 'y-', 'LineWidth', 1.5);

    % mark the intersection points
    plot(P1(1), P1(2), 'yx', 'LineWidth', 2);
    plot(P2(1), P2(2), 'yx', 'LineWidth', 2);

    % save
    P1_list(end+1,:) = P1;
    P2_list(end+1,:) = P2;
    theta_list(end+1,1) = th;
end

hold off;

% Homogeneous coordinates (3xM)
P1_h = [P1_list, ones(size(P1_list,1),1)]';
P2_h = [P2_list, ones(size(P2_list,1),1)]';

%% Computing projected midpoints using harmonic Cross-Ratio
% ==========================================================================
% Exploit the harmonic Cross-Ratio relation to find the image of midpoints
% between corresponding symmetric pairs of the arcs.
% The midpoint satisfies:
% cross-ratio(A,B;M,V) = -1
% and this condition is invariant under projective transformations.
% Here:
% - A and B are the two intersections P1 and P2 for the same direction θ
% - V is the vanishing point of the cylinder axis direction (V_axis)
% - M is the mid-point we wish to find
% ==========================================================================
Vh = V_axis(:);
Mnum = size(P1_h,2);

% Inhomogeneous endpoints
A = P1_h(1:2,:)./P1_h(3,:);
B = P2_h(1:2,:)./P2_h(3,:);

% Preallocate
M_xy = zeros(2, Mnum);

V = Vh(1:2)./Vh(3);

for i = 1:Mnum
        Ai = A(:,i);
        Bi = B(:,i);

        % Direction vector along the chord AB
        d  = Bi - Ai;

        dd = d.'*d;
        % Degenerate case: A and B coincide (or are numerically identical).
        % This could happen close to the intersection nodal point N_ij
        if dd < 1e-18
            M_xy(:,i) = Ai;
            continue;
        end

        % Parameterize the chord line by:
        % P(t) = A + t (B - A)
        % so that t=0 -> A,  t=1 -> B.
        %
        % Compute t_V: the parameter value where the vanishing point projects
        % onto the line AB.
        tV = ((V - Ai).'*d)/dd;

        % From harmonic condition CR(A,B;M,V) = -1, one obtains:
        %    t_M = t_V / (2 t_V - 1)
        denom = 2*tV - 1;
        if abs(denom) < 1e-12
            % numerically unstable case;
            warning('Pair %d: denom ~ 0 (2*tV-1).', i);
        end

        % Harmonic division parameter for the midpoint image M
        tM = tV/denom;

        % point M
        M_xy(:,i) = Ai + tM*d;
end

% Homogeneous output 3xM
M_h = [M_xy; ones(1,Mnum)];

figure(5);
imshow(I); hold on;

% plot symmetric couples
plot(P1_list(:,1), P1_list(:,2), 'yx', 'LineWidth', 2);
plot(P2_list(:,1), P2_list(:,2), 'yx', 'LineWidth', 2);

% plot harmonic points M
plot(M_xy(1,:), M_xy(2,:), 'ro', 'MarkerSize', 6, 'LineWidth', 2);

% connect P1-M-P2 for visual check
for i = 1:Mnum
    plot([P1_list(i,1) M_xy(1,i) P2_list(i,1)], ...
         [P1_list(i,2) M_xy(2,i) P2_list(i,2)], 'r-', 'LineWidth', 1);
end

hold off;

%% Fit an ellipse through the midpoint set M_xy
% =========================================================================
% Fit a 2D conic to the set of projected midpoints M_xy, and enforce that the
% fitted conic is specifically an ellipse.
% Using an approach where we fit multiple ellipses and select the best one
% through least-squares:
% minimize ||D * p||^2   subject to 4AC - B^2 = 1 (up to scale)
% where p = [A B C D E F]^T are the conic coefficients of:
% A x^2 + B x y + C y^2 + D x + E y + F = 0
%
% The constraint 4AC - B^2 > 0 is the discriminant condition for an ellipse
% (as opposed to hyperbola/parabola) in the quadratic part.
% =========================================================================
x = M_xy(1,:).';
y = M_xy(2,:).';

% normalization for numeric stability
mx = mean(x); my = mean(y);
sx = std(x);  sy = std(y);
xN = (x - mx) / sx;
yN = (y - my) / sy;

% Matrix for conic
D = [xN.^2, xN.*yN, yN.^2, xN, yN, ones(size(xN))];

% Scatter matrix
S = D' * D;

% Constraint matrix for ellipse-specific fit
% Enforces 4AC - B^2 = 1 (up to scale) in the generalized eigensystem
C = zeros(6);
C(1,3) = 2;
C(2,2) = -1;
C(3,1) = 2;

% Solve generalized eigenvalue problem S*a = lambda*C*a
[Vec, Val] = eig(S, C);

% Pick eigenvector that satisfies ellipse condition 4AC - B^2 > 0
disc = 4*Vec(1,:).*Vec(3,:) - Vec(2,:).^2;
idx = find(disc > 0 & isfinite(diag(Val)).', 1, 'first');

pN = Vec(:, idx); % conic parameters in normalized coordinates: [A B C D E F]

% Un-normalize: convert conic from (xN,yN) back to original (x,y)
A = pN(1); B = pN(2); Cc = pN(3); Dd = pN(4); Ee = pN(5); Ff = pN(6);

% Transform back by substitution: xN=(x-mx)/sx, yN=(y-my)/sy
A0 = A/(sx^2);
B0 = B/(sx*sy);
C0 = Cc/(sy^2);
D0 = Dd/sx - 2*A*mx/(sx^2) - B*my/(sx*sy);
E0 = Ee/sy - 2*Cc*my/(sy^2) - B*mx/(sx*sy);
F0 = Ff ...
    + A*(mx^2)/(sx^2) + B*(mx*my)/(sx*sy) + Cc*(my^2)/(sy^2) ...
    - Dd*mx/sx - Ee*my/sy;

p = [A0 B0 C0 D0 E0 F0].';   % final conic parameters in original coords

% Plot the fitted ellipse over the image and within image bounds
A = p(1); B = p(2); Cc = p(3); Dd = p(4); Ee = p(5); Ff = p(6);

figure(6); imshow(I); hold on;
f = @(X,Y) A*X.^2 + B*X.*Y + Cc*Y.^2 + Dd*X + Ee*Y + Ff;

xmin = 1; xmax = size(I,2);
ymin = 1; ymax = size(I,1);
fimplicit(f, [xmin xmax ymin ymax], 'g', 'LineWidth', 2);

hold off;

%% Final metric rectification
% Conic matrix in image coordinates
C = [ A    B/2  Dd/2;
      B/2  Cc   Ee/2;
      Dd/2 Ee/2 Ff ];

% Push conic through affine rectification: C_A = H_A^{-T} C H_A^{-1}
invHA = HA \ eye(3);       
C_A   = invHA.' * C * invHA;

% Quadratic part in the affine plane
Q = (C_A(1:2,1:2) + C_A(1:2,1:2).')/2;

% Ensure positive definite up to sign (ellipse => definite)
e = eig(Q);
if all(e < 0)
    Q = -Q;
    e = -e;
end

% Scale normalization
Q = Q / trace(Q);

% Metric upgrade: find A_M such that A_M^T Q A_M = I
R   = chol(Q,'upper');      % Q = R'R
A_M = R;           
HM = [A_M [0;0];
       0 0  1];

% Compute total rectification matrix H
HR = HM*HA;
tform = projective2d(HR.');

[I_rect, Rrect] = imwarp(I, tform); 

figure(7); imshow(I_rect, Rrect); axis image;
xlim([-2850 -1300]);
ylim([-600 50]);
title('Final metric-rectified image');

% sanity check: ellipse should become a circle in metric plane
invHR = HR \ eye(3);
C_rect = invHR.' * C * invHR;
Qm = (C_rect(1:2,1:2) + C_rect(1:2,1:2).')/2;
Qm = Qm / trace(Qm);
disp('Q in metric plane (should be ~0.5*I):');
disp(Qm);

%% Save final rectified image
imwrite(I_rect, fullfile('results', '4a-rectified_full.png'));

%% 3C - Position of non-apical point N_{ij} on the rectified plane in pixel coordinates
N_ij_rect = HR * N_ij;
N_ij_rect = N_ij_rect/N_ij_rect(3);

disp('Coordinates of non-apical nodal point N_ij in the rectified plane are:');
disp(N_ij_rect);

%% 4 - CAMERA CALIBRATION
% =========================================================================
% Unknown: omega = K^{-T}K^{-1} (IAC), symmetric 3x3, defined up to scale.
% Calibration from rectified face + orthogonal vanishing point:
%  - Use H_Z = inv(H_R) (plane->image homography)
%  - Use Zhang constraints (2 eqs): h1^T w h2 = 0,  h1^T w h1 - h2^T w h2 = 0
%  - Use l'_inf = w v (2 eqs): l x (w v) = 0
%  - Solve for w = K^{-T}K^{-1} (zero skew)
%  - Recover K from w
% =========================================================================

H_Z = inv(HR);
H_Z = H_Z / norm(H_Z(:,1));     % normalize scale

h1 = H_Z(:,1);
h2 = H_Z(:,2);

v = V_axis(:);
l = vanishingLine(:);

% solve for omega = K^{-T}K^{-1} under zero skew
% omega = [w11  0   w13;
%           0  w22  w23;
%          w13 w23  w33]
% unknown vector w = [w11 w22 w13 w23 w33]^T (scale-ambiguous)

row_vtWv = @(p,q) [ ...
    p(1)*q(1), ...                         % w11
    p(2)*q(2), ...                         % w22
    p(1)*q(3)+p(3)*q(1), ...               % w13
    p(2)*q(3)+p(3)*q(2), ...               % w23
    p(3)*q(3) ];                           % w33

A = []; % in this matrix we stack the 4 constraints

% Zhang constraints
A = [A; row_vtWv(h1,h2)];
A = [A; row_vtWv(h1,h1) - row_vtWv(h2,h2)];

% l x (omega v) = 0
% omega*v components (linear in w):
% u1 = w11*v1 + w13*v3
% u2 = w22*v2 + w23*v3
% u3 = w13*v1 + w23*v2 + w33*v3

v1=v(1); v2=v(2); v3=v(3);
l1=l(1); l2=l(2); l3=l(3);

% c1 = l2*u3 - l3*u2 = 0
A = [A; ...
    0, ...                          % w11
    -l3*v2, ...                     % w22
    l2*v1, ...                      % w13
    l2*v2 - l3*v3, ...              % w23
    l2*v3 ];                        % w33

% c2 = l3*u1 - l1*u3 = 0
A = [A; ...
    l3*v1, ...                      % w11
    0, ...                          % w22
    l3*v3 - l1*v1, ...              % w13
    -l1*v2, ...                     % w23
    -l1*v3 ];                       % w33

% Solve homogeneous least-sqaures via SVD
[~,~,Vsvd] = svd(A);
w = Vsvd(:,end);

omega = [ w(1)  0    w(3);
          0     w(2) w(4);
          w(3)  w(4) w(5) ];
omega = (omega + omega')/2; %force symmetry

% Fix global sign (omega should be PD up to scale)
if omega(1,1) < 0 && omega(2,2) < 0
    omega = -omega;
end
omega = (omega + omega')/2;

disp('H_Z:');   disp(H_Z);
disp('omega (unnormalized):'); disp(omega);

% Recover K:
% Scale-invariant principal point:
u0 = -omega(1,3)/omega(1,1);
v0 = -omega(2,3)/omega(2,2);

% Fix omega scale so that K(3,3)=1 (more stable):
% omega33 = u0^2*omega11 + v0^2*omega22 + 1
den = omega(3,3) - (u0^2)*omega(1,1) - (v0^2)*omega(2,2);
if den <= 0
    omega = -omega;
    den = omega(3,3) - (u0^2)*omega(1,1) - (v0^2)*omega(2,2);
end
lambda = 1 / den;
omegaN = lambda * omega;

fx = sqrt(1/omegaN(1,1));
fy = sqrt(1/omegaN(2,2));

K = [fx 0 u0;
     0  fy v0;
     0  0  1];

disp('Recovered K:');
disp(K);

% Residual checks (should be close to zero)
z1 = h1.'*omega*h2;
z2 = h1.'*omega*h1 - h2.'*omega*h2;
line_res = norm(cross(l, omega*v));
fprintf('Residuals: z1=%.3e, z2=%.3e, line=%.3e\n', z1, z2, line_res);

%% 5 - 3D COORDINATES OF OGIVAL RIB POINTS
%% Fixing the scale
figure(8); imshow(I_rect, Rrect); axis image; hold on
xlim([-2850 -1300]);
ylim([-600 50]);

% First point already known:
u1 = N_ij_rect(1);
v1 = N_ij_rect(2);
plot(u1, v1, 'r+', 'MarkerSize', 6, 'LineWidth', 2)

% Pick second point on the rectified view (which is also chosen as origin
% of the reference frame)
[x2, y2] = ginput(1);
u2 = x2; 
v2 = y2;
plot(u2, v2, 'r+', 'MarkerSize', 6, 'LineWidth', 2)

d_rect = hypot(u2 - u1, v2 - v1);

scale = 1 / d_rect % d = 1

%% Building Vault Cylinder
% Cylinder axis direction in 3D
v = V_axis(:);                 % 3x1 homogeneous
a = (K \ v);                    % inv(K)*v
a = a / norm(a);

% extract circle parameters for centre and radius computation
A = C_rect(1,1);
B = 2*C_rect(1,2);
Cq = C_rect(2,2);
D = 2*C_rect(1,3);
E = 2*C_rect(2,3);
F = C_rect(3,3);

M = [2*A, B; B, 2*Cq];
c = -M \ [D; E];         % [x0; y0]
x0 = c(1); y0 = c(2);  % CENTER
val = A*x0^2 + B*x0*y0 + Cq*y0^2 + D*x0 + E*y0 + F;
r2  = -val / A;
r_plane = sqrt(r2); % RADIUS

% Scaling
R_cyl = scale * r_plane;                 % metric radius
center_plane_metric = scale * [x0; y0];  % metric center in plane coords

% Compute R,t of rectified plane w.r.t. camera
H_p2i = H_Z;   % plane -> image
Bhom = K \ H_p2i;              % inv(K)*HR
b1 = Bhom(:,1); b2 = Bhom(:,2); b3 = Bhom(:,3);

s  = 1 / ((norm(b1) + norm(b2))/2);   % scale for rotation columns
r1 = s*b1;
r2 = s*b2;
r3 = cross(r1, r2);
t  = s*b3;
t = scale * t; %fix to global scale

R = [r1 r2 r3];

% Orthonormalize R
[U,~,V] = svd(R);
R = U*V';

if det(R) < 0
    U(:,3) = -U(:,3);
    R = U*V';
end

% Ensure r3 points along cylinder axis direction a (sign consistency)
if dot(R(:,3), a) < 0
    R(:,1:2) = -R(:,1:2);
    R(:,3)   = -R(:,3);
    t        = -t;
end


% Lift center to 3D point in camera reference
X0_plane = [center_plane_metric; 0]; 
c0 = R * X0_plane + t; % point on cylinder axis in camera frame


% Visualize 3D cylinder
L = 4 * R_cyl;         % half-length of the cylinder
nTheta = 80;           % around
nZ = 40;               % along axis

% Build an orthonormal basis {u, v, a}:
% pick any vector not parallel to a
tmp = [1;0;0];
if abs(dot(tmp,a)) > 0.9
    tmp = [0;1;0];
end

u = cross(a, tmp);
u = u / norm(u);
v = cross(a, u);    % already unit if a,u unit and orthogonal

% Parametrize cylinder surface
theta = linspace(0, 2*pi, nTheta);
z = linspace(-L, L, nZ);
[TH, ZZ] = meshgrid(theta, z);

X = c0(1) + a(1)*ZZ + R_cyl*(u(1)*cos(TH) + v(1)*sin(TH));
Y = c0(2) + a(2)*ZZ + R_cyl*(u(2)*cos(TH) + v(2)*sin(TH));
Z = c0(3) + a(3)*ZZ + R_cyl*(u(3)*cos(TH) + v(3)*sin(TH));

% Plot in camera frame (origin = camera)
figure(10); clf; hold on; grid on; axis equal; axis vis3d;

surf(X, Y, Z, 'FaceAlpha', 0.25, 'EdgeColor', 'none');

% mark camera
plot3(0,0,0,'b.','MarkerSize',20);
text(0,0,0,'  camera','Color','b','FontWeight','bold');

xlabel('X_c'); ylabel('Y_c'); zlabel('Z_c');
title('Cylinder in camera coordinates');

% --- Make it visible without changing geometry ---
pad = 6*R_cyl;
camtarget(c0');
view(3);

rotate3d on; pan on; zoom on;

%%  Selection of 12 points on the rib
figure(9); imshow(I); axis on; hold on;

% Selection of ogival rib of family a_i: manually sample points along the rib crossing N_{ij}.
Kpts = 12;
title('Select 12 points on the ogival rib crossing N_{ij} of family a_i');
rib = zeros(3, Kpts);
for k = 1:Kpts
    [x1,y1] = ginput(1);
    rib(:,k) = [x1; y1; 1]; %homogeneus coordinates
    plot(x1,y1,'ro','MarkerSize',8,'LineWidth',2);
    text(x1+5,y1, sprintf('a_{%d}',k), ...
         'Color','r','FontSize',12,'FontWeight','bold');
end
disp('Press enter to continue');

%% Intersection between camera rays and cylinder
% Compute camera rays of selected points
Kinv = inv(K);
nPts = size(rib,2);

ray_dirs = zeros(3,nPts);

for k = 1:nPts
    d = Kinv * rib(:,k);   % backproject pixel
    d = d / norm(d);      % normalize direction
    ray_dirs(:,k) = d;
end

cam_center = [0;0;0];     % camera origin

% Intersect rays with cylinder
rib_3D = zeros(3,nPts);

for k = 1:nPts
    o = cam_center;
    d = ray_dirs(:,k);

    % components perpendicular to axis
    d_perp = d - a*(a.'*d);
    oc = o - c0;
    oc_perp = oc - a*(a.'*oc);

    A = dot(d_perp, d_perp);
    B = 2 * dot(d_perp, oc_perp);
    C = dot(oc_perp, oc_perp) - R_cyl^2;

    disc = B^2 - 4*A*C;

    if disc < 0
        warning('No intersection for ray %d', k);
        rib_3D(:,k) = NaN;
        continue;
    end

    % two intersections
    lambda1 = (-B + sqrt(disc)) / (2*A);
    lambda2 = (-B - sqrt(disc)) / (2*A);

    % choose the closest *in front* of the camera
    lambdas = [lambda1 lambda2];
    lambdas = lambdas(lambdas > 0);

    if isempty(lambdas)
        warning('Intersection behind camera for ray %d', k);
        rib_3D(:,k) = NaN;
        continue;
    end

    lambda = max(lambdas);
    rib_3D(:,k) = o + lambda * d;
end

figure(10); hold on;
plot3(rib_3D(1,:), rib_3D(2,:), rib_3D(3,:), ...
      'ro-', 'LineWidth', 2, 'MarkerSize', 8);

camzoom(0.65)

%% sanity check
for k = 1:size(rib_3D,2)
    X = rib_3D(:,k);
    Xaxis = c0 + a * dot(a, (X - c0));
    r = norm(X - Xaxis);
    fprintf('%2d: ||X||=%.3f  radius=%.3f\n', k, norm(X), r);
end

%% 6 - LOCALIZATION OF CYLINDER AXIS
% I already have what's needed: a point on the axis (c0) and it's direction
% (a).
% Adding cylinder axis to the plot
Laxis = 4 * R_cyl;

% axis endpoints
A1 = c0 - Laxis * a;
A2 = c0 + Laxis * a;

figure(10); hold on;

% plot axis line
plot3([A1(1) A2(1)], ...
      [A1(2) A2(2)], ...
      [A1(3) A2(3)], ...
      'm-', 'LineWidth', 3);

plot3(c0(1), c0(2), c0(3), ...
      'mo', 'MarkerSize', 10, 'LineWidth', 2);

% label
text(c0(1)-0.5, c0(2), c0(3)-1, ...
     '  axis', 'Color', 'm', 'FontSize', 12, 'FontWeight', 'bold');

% display axis parameters
a = a(:);
fprintf('Direction of the cylinder axis a = [%.6f %.6f %.6f]\n', a(1), a(2), a(3));
fprintf('Point on the cylinder axis c0 = [%.6f %.6f %.6f]\n', c0(1), c0(2), c0(3));
