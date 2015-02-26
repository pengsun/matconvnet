classdef tfw_i < tf_i
  %TFW_I Transformer Wrapper Interface (Base Class)
  %   Detailed explanation goes here
  
  properties
    tfs; % for internal transformer array
  end
  
  methods % auxiliary
    
    function ob = cl_io (ob)
      ob = cl_io@tf_i(ob);
      ob.tfs = cellfun(@cl_io, ob.tfs, 'uniformoutput',false);
    end
    
  end % methods auxiliary
  

end

