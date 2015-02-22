classdef opt_1storder < opt_i
  %OPT_1STORDER First Order Method
  %   Detailed explanation goes here
  
  properties
    mo; % momentum
    wd; % weight decay
    eta; % step size
    
    delta; % delta at last iteration
  end
  
  methods
    function obj = opt_1storder(obj, varargin)
      
    end
    
    function obj = update(obj, pa)
      pa.a = pa.a - 0.00001 * pa.d;
    end
  end
  
end

