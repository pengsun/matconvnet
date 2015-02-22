classdef tf_i
  %TF_I Transformer Interface (Base Class)
  %   Detailed explanation goes here
  
  properties
    p; % parameters
    i; % input variables
    o; % output variables
  end
  
  methods
    function obj = fprop(obj)
    end
    
    function obj = bprop(obj)
    end
    
  end
  
end

