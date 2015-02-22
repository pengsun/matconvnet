classdef opt_i
  %PARAMOPT_I Parameter Optimizer Interface (Base Class)
  %   Detailed explanation goes here
  
  properties
    cc; % calling context
  end
  
  methods
    function obj = paramopt_i(obj)
      obj.cc = struct();
    end
    
    function obj = update(obj, pa)
      error('Should not be called in base class\n');
    end
    
  end
  
end

