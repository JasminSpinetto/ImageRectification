%% Image Analysis and Computer Vision Homework by Jasmin Spinetto
% Year 2025/26

% This code provides a function to extract relevant segment features from the San
% Maurizio picture. 

% It is handled separately so to save the extracted segments in a static .mat file
% from which one can extract them in the main code. This way,
% features will be stable when developing the main code.

clear all 
close all
clc

% Load the image for analysis
I = imread(fullfile('data', 'raw', 'san_maurizio.jpg'));

FNT_SZ = 20;

%% Select relevant vertical lines manually
figure(1); imshow(I); hold on;
title('Draw 3 vertical lines:');

numSegments_v = 3;
endPointsV = zeros(numSegments_v,4);   % [x1 y1 x2 y2] save only endpoints of each line

count = 1;
while count <= numSegments_v
    seg = drawline('StripeColor','black');

    endPointsV(count,:) = [seg.Position(1,:) seg.Position(2,:)];
    count = count + 1;
end

fprintf('Press enter to continue\n');

hold off;

%% Select relevant horizontal lines manually
figure(1); hold on;
title('Draw 3 horizontal lines:');

numSegments_h = 3;
endPointsH = zeros(numSegments_h,4);   % [x1 y1 x2 y2]

count_h = 1;
while count_h <= numSegments_h
    seg = drawline('StripeColor','white');

    endPointsH(count_h,:) = [seg.Position(1,:) seg.Position(2,:)];
    count_h = count_h + 1;
end

fprintf('Press enter to continue\n');

hold off;

%% Select lines parallel to cylinder axis manually
figure(1); hold on;
title('Draw 4 axis parallel lines:');

numSegments_ax = 4;
endPointsAX = zeros(numSegments_ax,4);   % [x1 y1 x2 y2]

count_ax = 1;
while count_ax <= numSegments_ax
    seg = drawline('StripeColor','green');

    endPointsAX(count_ax,:) = [seg.Position(1,:) seg.Position(2,:)];
    count_ax = count_ax + 1;
end

fprintf('Press enter to continue\n');

hold off;

%% Save to file
save(fullfile('data', 'processed', 'san_maurizio_features.mat'), ...
     'N_ij', ...
     'symP1', 'symP2', ...
     'endPointsV', ...
     'endPointsH', ...
     'endPointsAX');

fprintf('Features saved to san_maurizio_features.mat\n');



