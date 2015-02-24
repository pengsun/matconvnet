classdef tf_dropout < tf_i
  %TF_DROPOUT Dropout
  %   Detailed explanation goes here
  
  properties
    %is_disabled;
    %is_freezed;
    rate; 
    
    mask;
  end
  
  methods
    function ob = tf_dropout ()
      ob.rate = 0.5;
    end
    
    function ob = fprop(ob)
      if ( ob.cc.is_tr ) % training stage: multiply a random mask
        [ob.o.a, ob.mask] = vl_nndropout(ob.i.a, 'rate',ob.rate);
      else % testing: multiply a scalar rate
        ob.o.a = ob.i.a;
      end
    end % fprop
    
    function ob = bprop(ob)
      if ( ob.cc.is_tr ) % training stage: multiply a random mask
        ob.i.d = vl_nndropout(ob.i.a, ob.o.d, 'mask',ob.mask);
      else % testing
        ob.i.d = ob.o.d; % ?
      end
    end % bprop
    
  end
  
end

