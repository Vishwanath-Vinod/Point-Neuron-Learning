function bias = structured_bias_mesh(sources,MicCoord,z)
    source1 = sources(1,:); 
    source2 = sources(2,:);
    % Dimensions of the volume (in meters)
    Lx = 1.3;
    Ly = 1.6;

    % Target region center and radius
    target_center   = [0.65,0.80,0.525];
    target_radius   = 0.50;

    % Resolutions
    res_x = 0.26; %0.13; %0.26;
    res_y = 0.4; %0.2;

    % Construct axis vectors without exceeding room size
    x = 0:res_x:Lx;  x = x(x <= Lx);  % max 10.4
    y = 0:res_y:Ly;  y = y(y <= Ly);  % max 10.4

    % Generate 3D grid
    [X, Y, Z] = meshgrid(x, y, z);

    % Flatten to N x 3 matrix
    bias = [X(:), Y(:), Z(:)];
    origin_idx = all(bias == 0, 2);
    bias(origin_idx, :) = [];
    % Remove points inside target spherical region
    distances = sqrt(sum((bias - target_center).^2, 2));
    bias = bias(distances > target_radius, :);
    min_dist = 0.05;
    dist_to_source = sqrt(sum((bias - source1).^2, 2));
    bias = bias(dist_to_source > min_dist, :);
    dist_to_source2 = sqrt(sum((bias - source2).^2, 2));
    bias = bias(dist_to_source2 > min_dist, :);

    % Remove points too close to any microphone
    if ~isempty(MicCoord)
        to_keep = true(size(bias, 1), 1);
        for i = 1:size(MicCoord, 1)
            dist_to_mic = sqrt(sum((bias - MicCoord(i, :)).^2, 2));
            to_keep = to_keep & (dist_to_mic > min_dist);
        end
        bias = bias(to_keep, :);
    end
    disp(size(bias));
end