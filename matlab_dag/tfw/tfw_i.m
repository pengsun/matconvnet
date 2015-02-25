classdef tfw_i < tf_i
  %TFW_I Transformer Wrapper Interface (Base Class)
  %   Detailed explanation goes here
  
  properties
    tfs; % for internal transformer array
  end
  
  methods
    function ob = fprop(ob)
      ob.tfs = cellfun(@fprop, ob.tfs, 'uniformoutput',false);
    end % fprop
    
    function ob = bprop(ob)
      ob.tfs(end:-1:1) = cellfun(@bprop, ob.tfs(end:-1:1),...
        'uniformoutput',false);
    end % bprop
    
  end % methods
  
end

