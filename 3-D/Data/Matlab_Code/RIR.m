clear 
rng('default');
rng(1);
L               = [1.3,1.6,1.05]; % Room dimensions
beta            = [0.8,0.8,0.8,0.8,0.8,0.8]; % Reflection coefficients
Frequency       = 900; % Frequency
c               = 343; % Speed of sound
k               = 2*pi*Frequency/c; % Wave Number

% Target region center and radius
target_center   = [0.65,0.8,0.525];
target_radius   = 0.50;


num_sources = 5;
sources = zeros(num_sources, 3);  % Preallocate
count = 0;

while count < num_sources
    % Generate a random point in the room
    candidate = [rand() * L(1), rand() * L(2), rand() * L(3)];
    
    % Compute distance from target center
    dist = norm(candidate - target_center);
    
    % Accept only if it's OUTSIDE the target region
    if dist > target_radius
        count = count + 1;
        sources(count, :) = candidate;
    end
end

disp(sources);

% Number of microphones based on frequency Q=2*Ns+1
Ns              = k*target_radius;
num_mics        = ((ceil(Ns)+1)^2) ;
% Allot microphones randomly within the spherical target region and take readings

% Allot microphones randomly within the spherical target region and take readings
MicCoord = sample_spherical_region(target_center, target_radius, num_mics);
MicData = zeros(size(MicCoord, 1), 1);  
for i = 1:size(sources, 1)
    MicData = MicData + ism(k, MicCoord, sources(i, :), L, beta, 10);
end
MicData = awgn((MicData),20,'measured');

%Sample pressure readings in the target region
num_samples_target = 1500;
sample_center = [0.65,0.8,0.525];
sampling_target = sample_circular_region(sample_center, target_radius, num_samples_target);
pressure_target = zeros(size(sampling_target, 1), 1);  
for i = 1:size(sources, 1)
    pressure_target = pressure_target + ism(k, sampling_target, sources(i, :), L, beta, 10);
end

%Sample pressure readings in the whole room

% Spatial resolution
dx = 0.025;
dy = 0.025;
dz = 0;

% Define grid points along each axis
x = 0:dx:L(1);
y = 0:dy:L(2);
z = 0.525;
% Generate 3D meshgrid
[X, Y, Z] = meshgrid(x, y, z);
sampling_room = [X(:), Y(:), Z(:)];
pressure_room = zeros(size(sampling_room, 1), 1);  
for i = 1:size(sources, 1)
    pressure_room = pressure_room + ism(k, sampling_room, sources(i, :), L, beta, 10);
end
disp(size(pressure_room));
num_zeros = sum(real(pressure_room) == 0 & imag(pressure_room) == 0);
disp(['Number of zero complex entries in pressure_room: ', num2str(num_zeros)]);

% Number of Point Neurons
V = 126;
% Initialize weights
angles = 2*pi*rand(V,1);
weights = (2*rand(V,1)-1) .* exp(1j * angles);
% Initialize PN locations (biases)
%biases = [];
%for i = 1:size(sources, 1)
%    z_i = sources(i, 3);  % extract z-coordinate of the i-th source
%    bias_i = structured_bias_mesh(sources, MicCoord, z_i);  % call with z_i
%    biases = [biases; bias_i];  % vertically concatenate
%end
biases = [structured_bias_mesh(sources,MicCoord,0.2);structured_bias_mesh(sources,MicCoord,0.4);
          structured_bias_mesh(sources,MicCoord,0.6);structured_bias_mesh(sources,MicCoord,0.8);
          structured_bias_mesh(sources,MicCoord,1.0)];
disp(size(biases))
% Save mic positions and mic signals
save('MIC_DATA_2D.mat', 'MicCoord', 'MicData');
save('P_TARGET_2D.mat', 'sampling_target', 'pressure_target');
save('P_ROOM_2D.mat', 'sampling_room', 'pressure_room');
save('PN_INITIALIZATION_2D.mat', 'weights', 'biases');
% Visualization
figure;
scatter3(MicCoord(:,1), MicCoord(:,2), MicCoord(:,3), 40, 'filled', 'MarkerFaceColor', 'blue');
hold on;
scatter3(sources(:,1), sources(:,2), sources(:,3), 10, 'red', 'filled');
scatter3(biases(:,1), biases(:,2), biases(:,3), 40, 'kx');  % 'k' = black, 'x' = cross
hold off;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Microphones, Point Neurons and Source');
legend('Microphones', 'Source', 'Point Neurons', 'Location', 'best');
axis equal; grid on;
xlim([0, L(1)]);
ylim([0, L(2)]);
zlim([0, L(3)]);
saveas(gcf, 'mic_source_pn_plot_2D.png');