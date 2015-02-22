classdef tf_relu < tf_i
  %TF_RELU RELU
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = fprop(ob)
      ob.o.a = vl_nnrelu(ob.i.a);
    end % fprop
    
    function ob = bprop(ob)
      ob.i.d = vl_nnrelu(ob.i.a, ob.o.d);
    end % bprop    
  end
  
end

