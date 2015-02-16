function Y = my_lse(X,c,dzdy)
% MY_LSE  least square error loss

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
%
% 2/16/2015: modified by pengsun000@gmail.com
%

%X = X + 1e-6 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

% reshape c the same size with x
if ( size(c,1)==sz(3) && size(c,2)==sz(4) )
  % one label per image
  c = reshape(c, [1 1 sz(3) sz(4)]) ;
  c = repmat(c, [sz(1) sz(2)]) ;
end

% compute 
delta = X - c;

if nargin <= 2 % fprop
  t = delta .* delta;
   % sum over all dimensions (sz(3)) and instances (sz(4))
  Y = 0.5 * sum(t(:));
else % bprop
  Y = delta .* dzdy;
end
