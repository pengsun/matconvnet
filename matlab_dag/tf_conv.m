classdef tf_conv < tf_i
  %TF_CONV Convolution
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = fprop(ob)
      w = ob.p(1).a;
      b = ob.p(2).a;
      ob.o.a = vl_nnconv(ob.i.a, w,b, 'pad',0, 'stride',1);
    end
    
    function ob = bprop(ob)      
      w = ob.p(1).a;
      b = ob.p(2).a;
      delta = ob.o.d;
      [ob.i.d, ob.p(1).d, ob.p(2).d] = vl_nnconv(...
        ob.i.a, w, b, delta, 'pad',0, 'stride',1);
    end
    
  end
  
end

