classdef tf_i
  %TF_I Transformer Interface (Base Class)
  %   Detailed explanation goes here
  
  properties
    p; % parameters
    i; % input variables
    o; % output variables
    
    cc; % calling context
  end
  
  methods
    function ob = tf_i()
      ob.cc = call_cntxt();
    end
    
    function ob = fprop(ob)
    end
    
    function ob = bprop(ob)
    end
    
  end
  
  methods % auxiliary
    function ob = cl_io(ob)
    % clear input, output data
      for k = 1 : numel(ob.i)
        ob.i(k).a = [];
        ob.i(k).d = [];
      end % for k
      for k = 1 : numel(ob.o)
        ob.o(k).a = [];
        ob.o(k).d = [];
      end % for k
    end % cl_io
  end % methods auxiliary
  
end

