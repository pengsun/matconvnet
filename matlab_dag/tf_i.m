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
      ob.cc = dag_util.create_cc();
    end
    
    function ob = fprop(ob)
    end
    
    function ob = bprop(ob)
    end
    
  end
  
end

