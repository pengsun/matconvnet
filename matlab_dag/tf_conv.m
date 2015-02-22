classdef tf_conv < tf_i
  %TF_CONV Convolution
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = fprop(ob)
      ob.o.a = ob.i(1).a .* randn(1,1);
    end
    
    function ob = bprop(ob)
      ob.i.d = ob.o.d .* randn(1,1);
      
      ob.p(1).d = ob.o.d .* randn(1,1);
      ob.p(2).d = ob.o.d .* randn(1,1);
    end
    
  end
  
end

