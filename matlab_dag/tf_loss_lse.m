classdef tf_loss_lse < tf_i
  %TF_LOSS_LSE Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = tf_loss_lse()
      ob.i = [n_data(), n_data()];
      ob.o = n_data();
    end
    
    function ob = fprop(ob)
      pre = squeeze(ob.i(1).a);
      tar = ob.i(2).a;
      
      ob.o.a = 0.5 * sum((pre - tar).^2, 1); 
    end
    
    function ob = bprop(ob)
      pre = squeeze(ob.i(1).a);
      tar = ob.i(2).a;
      
      ob.i(1).d = (pre - tar) .* ob.o.d;  
      ob.i(1).d = reshape(ob.i(1).d, size(ob.i(1).a) );
    end
    
  end
  
end

