classdef tf_loss_lse < tf_i
  %TF_LOSS_LSE Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = fprop(ob)
      ob.o.a = ob.i(1).a .* randn(1,1);
    end
    
    function ob = bprop(ob)
      ob.i(1).d = ob.o.d .* randn(1,1);
      
    end
    
  end
  
end

