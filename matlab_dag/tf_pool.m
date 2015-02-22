classdef tf_pool < tf_i
  %TF_POOL Pooling
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = fprop(ob)
      ob.o.a = vl_nnpool(ob.i.a, [2,2],...
        'pad',0, 'stride',2, 'method','max');
    end % fprop
    
    function ob = bprop(ob)
      ob.i.d = vl_nnpool(ob.i.a, [2,2], ob.o.d,...
        'pad',0, 'stride',2, 'method','max');
    end % bprop
  end % methods
  
end

