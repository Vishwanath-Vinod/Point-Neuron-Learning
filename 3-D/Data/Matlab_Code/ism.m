function [S] = ism(k, mic_xyz, source_xyz, room_size, room_b, room_order)
% [S] = ism(k, mic_xyz, source_xyz, room_size, room_b, room_order) 
% -------------------------------------------------------------------------
% Package   : SPHLIB.m
% Function  : ism
% Creation  : 10-09-18
% Author    : Hanchi Chen, Lachlan Birnie
% Version   : v1.10 (11-06-19)
%
% Description
%
%   Return image source simulated pressures at each microhpone position 
%   for a single given point source, room and frequency. 
%
% Inputs
%
%   k           : [scalar] wave number (frequency of source).
%   mic_xyz     : [Q by 3] vector of microphone positions (xq,yq,zq)
%                  w.r.t the room's bottom left corner.
%   source_xyz  : [1 by 3] vector of source positions (x,y,z) 
%                 w.r.t room's bottom left corner.
%   room_size   : [1 by 3] vector of (length,width,height) room dimensions
%                 in meters (m).
%   room_b      : [1 by 6] / [scalar] wall reflection coefficients.
%   room_order  : (D) reflection order / image depth of simulation.
%
% Outputs
%
%   S           [Q by 1] vector of microphone pressure measurements.
%
% Acknowledgements
%   
%   - Orginal code taken from Hanchi Chen, "frrespArrayInput.m"
%   - Lachlan removed the for-loops for matrix calculations.
%
% Equation
%                 1      D
%                ----  ---- 
%                \     \      b1^|r1-p1| * b2^|r1| * b3^|r2-p2| * b4^|r2| 
%   S(k,Xq,Xl) =  >     >       * b5^|r3-p3| * b6^|r3| * e^(ik |Rp + Rr|)
%                /     /                                ------------------
%                ----  ----                                4pi |Rp + Rr|
%               p = 0  r = -D
%               _____ _______                           __________________
%                 ^      ^                                      ^
%           These are tripple summations,               This is Green's
%           p = (p1,p2,p3),  r = (r1,r2,r3)             Function. 
%
%   Where:
%
%       b = wall reflections coefficients b = [b1,b2,b3,b4,b5,b6]. 
%
%       Xq = (xq,yq,zq) position of microphone.
%       Xl = (xl,yl,zl) position of source.
%   
%       Rp = (xq - xl + 2*p1*xl, yq - yl + 2*p2*yl, zq - zl + 2*p3*zl)
%       Rr = (2*r1*room_length, 2*r2*room_width, 2*r3*room_height)
%
% History
%
%   28-01-21    - Moved to SPHLIB from LIB. 
%
% -------------------------------------------------------------------------

    % Allow for single room_b value on all walls.
    if isscalar(room_b)
        room_b = repmat(room_b, [1 6]);
    end

    % p summations.
    p1 = [0, 0, 0, 0, 1, 1, 1, 1];
    p2 = [0, 0, 1, 1, 0, 0, 1, 1];
    p3 = [0, 1, 0, 1, 0, 1, 0, 1];

    Rp1 = source_xyz(1) + 2*mic_xyz(:,1)*p1  ...
            - repmat(mic_xyz(:,1), 1, length(p1));
        
    Rp2 = source_xyz(2) + 2*mic_xyz(:,2)*p2  ...
            - repmat(mic_xyz(:,2), 1, length(p2));
        
    Rp3 = source_xyz(3) + 2*mic_xyz(:,3)*p3  ...
            - repmat(mic_xyz(:,3), 1, length(p3)); % [Q by 8 by 1]

    % r summations [(2D+1)^3 by 1].
    D = room_order;
    
    r1 = repmat((-D : D).', [1 ,2*D+1, 2*D+1]); 
    r2 = repmat((-D : D), [2*D+1, 1, 2*D+1]);
    r3 = repmat(permute((-D : D).', [3,2,1]), [2*D+1, 2*D+1, 1]); 

    r1 = r1(:); % [(D+1)^2,(D+1)^2,(D+1)^2] --> [(2D+1)^3, 1]
    r2 = r2(:);
    r3 = r3(:);
    
    % Reflection coefficients.
    bc = room_b(1).^(abs(r1 - p1)) ...
      .* room_b(2).^(abs(r1)) ...
      .* room_b(3).^(abs(r2 - p2)) ...
      .* room_b(4).^(abs(r2)) ...
      .* room_b(5).^(abs(r3 - p3)) ...
      .* room_b(6).^(abs(r3));  % [1,8]
    
    bc = permute(bc.', [3,1,2]); %[1,1,8]

    % Find path distance.
    Rr1 = 2 .* r1 .* room_size(1);
    Rr2 = 2 .* r2 .* room_size(2);
    Rr3 = 2 .* r3 .* room_size(3);  % [(2D+1)^3 by 1]

    Rr1 = permute(Rr1, [2,3,1]);
    Rr2 = permute(Rr2, [2,3,1]);
    Rr3 = permute(Rr3, [2,3,1]);  % [1 by 1 by (2D+1)^3]
    
    % [(D+1)^2, (D+1)^2, (D+1)^2]
    Rd = sqrt( (Rp1 + Rr1).^2 + (Rp2 + Rr2).^2 + (Rp3 + Rr3).^2 );

    % Solve for pressure [Q by 1]:
    S = exp((1i*k) .* Rd) ./ ((4*pi) .* Rd) .* bc;  % [Q by 8 by (2D+1)^3]
    S = sum(S,2);  % [Q,1,D]
    S = sum(S,3);  % [Q,1,1] --> [Q]

end