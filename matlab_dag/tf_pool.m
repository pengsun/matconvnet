classdef tf_pool < tf_i
  %TF_POOL Pooling
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = fprop(ob)
      ob.o.a = ob.i.a .* randn(1,1);
    end
    
    function ob = bprop(ob)
      ob.i.d = ob.o.d .* randn(1,1);
    end
    
  end
  
end

